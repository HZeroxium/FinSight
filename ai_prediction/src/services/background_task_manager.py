# services/background_task_manager.py

"""
Background task manager for handling asynchronous training jobs
"""

import asyncio
import uuid
import time
import psutil
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..schemas.training_schemas import (
    AsyncTrainingRequest,
    TrainingJobInfo,
    TrainingProgressUpdate,
    TrainingJobPriority,
    BackgroundTaskHealth,
)
from ..core.constants import TrainingJobStatus, TrainingConstants, BackgroundTaskConfig
from ..repositories.training_job_facade import (
    TrainingJobFacade,
)
from common.logger.logger_factory import LoggerFactory
from ..core.config import get_settings


@dataclass
class TaskWorker:
    """Task worker information"""

    worker_id: str
    is_busy: bool = False
    current_job_id: Optional[str] = None
    started_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    processed_jobs: int = 0


class BackgroundTaskManager:
    """
    Manages background training tasks with proper concurrency control,
    queue management, and resource monitoring
    """

    def __init__(self, job_facade: TrainingJobFacade):
        self.logger = LoggerFactory.get_logger("BackgroundTaskManager")
        self.settings = get_settings()
        self.job_facade = job_facade

        # Worker management
        self.workers: Dict[str, TaskWorker] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=TrainingConstants.MAX_CONCURRENT_TRAININGS,
            thread_name_prefix="training_worker",
        )

        # Queue management
        self.job_queue: asyncio.Queue = asyncio.Queue(
            maxsize=TrainingConstants.MAX_QUEUE_SIZE
        )
        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Task management
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.cancelled_jobs: set[str] = set()

        # Performance tracking
        self.start_time = datetime.now(timezone.utc)
        self.jobs_processed = 0
        self.jobs_failed = 0
        self.total_processing_time = 0.0

        # Health monitoring
        self.last_health_check = datetime.now(timezone.utc)
        self.is_healthy = True
        self.health_issues: List[str] = []

        # Background tasks
        self._manager_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_initialized = False

        # Initialize workers (sync initialization)
        self._initialize_workers()

        # Don't start background tasks here - will be started later when event loop is ready

    async def initialize(self) -> None:
        """
        Initialize async components (call this when event loop is running)
        """
        if self._is_initialized:
            return

        try:
            # Start background tasks
            self._start_background_tasks()

            self._is_initialized = True
            self.logger.info("BackgroundTaskManager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize BackgroundTaskManager: {e}")
            raise

    async def _ensure_initialized(self) -> None:
        """Ensure manager is initialized before use"""
        if not self._is_initialized:
            await self.initialize()

    async def submit_job(
        self,
        request: AsyncTrainingRequest,
        training_function: Callable,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """
        Submit a training job to the background queue

        Args:
            request: Training request
            training_function: Function to execute the training
            progress_callback: Optional callback for progress updates

        Returns:
            str: Job ID

        Raises:
            ValueError: If queue is full or system is unhealthy
        """
        await self._ensure_initialized()

        try:
            # Check system health
            if not self.is_healthy:
                raise ValueError("Background task system is unhealthy")

            # Check queue capacity
            if self.job_queue.qsize() >= TrainingConstants.MAX_QUEUE_SIZE:
                raise ValueError("Training queue is full")

            # Create job ID
            job_id = str(uuid.uuid4())

            # Estimate completion time
            estimated_duration = self._estimate_training_duration(request)

            # Create job info
            job_info = TrainingJobInfo(
                job_id=job_id,
                symbol=request.symbol,
                timeframe=request.timeframe.value,
                model_type=request.model_type.value,
                status=TrainingJobStatus.PENDING,
                progress=0.0,
                created_at=datetime.now(timezone.utc),
                config=request.config.model_dump(),
                priority=request.priority,
                tags=request.tags,
                estimated_remaining_seconds=estimated_duration,
            )

            # Save job to repository
            await self.job_facade.create_job(job_info)

            # Create task data
            task_data = {
                "job_id": job_id,
                "request": request,
                "training_function": training_function,
                "progress_callback": progress_callback,
                "submitted_at": datetime.now(timezone.utc),
                "priority_score": self._calculate_priority_score(request.priority),
            }

            # Add to priority queue
            await self.priority_queue.put((task_data["priority_score"], task_data))

            # Update job status
            await self.job_facade.update_job(
                job_id,
                {
                    "status": TrainingJobStatus.QUEUED,
                    "progress": TrainingConstants.PROGRESS_STAGES[
                        TrainingJobStatus.QUEUED
                    ],
                },
            )

            self.logger.info(
                f"Submitted job {job_id} to queue (priority: {request.priority})"
            )
            return job_id

        except Exception as e:
            self.logger.error(f"Failed to submit job: {e}")
            raise

    async def cancel_job(self, job_id: str, force: bool = False) -> bool:
        """
        Cancel a training job

        Args:
            job_id: Job identifier
            force: Force cancellation even if in critical stage

        Returns:
            bool: True if job was cancelled successfully
        """
        try:
            # Get job info
            job_info = await self.job_facade.get_job(job_id)
            if not job_info:
                self.logger.warning(f"Job {job_id} not found for cancellation")
                return False

            # Check if job can be cancelled
            if not force and job_info.status in [TrainingJobStatus.SAVING_MODEL]:
                self.logger.warning(
                    f"Job {job_id} is in critical stage, use force=True to cancel"
                )
                return False

            # Add to cancelled set
            self.cancelled_jobs.add(job_id)

            # Cancel active task if running
            if job_id in self.active_tasks:
                task = self.active_tasks[job_id]
                task.cancel()

                # Wait for task to complete cancellation
                try:
                    await asyncio.wait_for(task, timeout=30.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

                # Remove from active tasks
                self.active_tasks.pop(job_id, None)

            # Update job status
            await self.job_facade.update_job(
                job_id,
                {
                    "status": TrainingJobStatus.CANCELLED,
                    "completed_at": datetime.now(timezone.utc),
                    "error_message": "Job cancelled by user",
                },
            )

            # Free up worker if assigned
            for worker in self.workers.values():
                if worker.current_job_id == job_id:
                    worker.is_busy = False
                    worker.current_job_id = None
                    worker.started_at = None
                    break

            self.logger.info(f"Cancelled job {job_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    async def get_queue_info(self) -> Dict[str, Any]:
        """Get information about the training queue"""
        try:
            active_jobs = await self.job_facade.get_active_jobs()

            # Count jobs by status
            pending_count = sum(
                1 for job in active_jobs if job.status == TrainingJobStatus.PENDING
            )
            queued_count = sum(
                1 for job in active_jobs if job.status == TrainingJobStatus.QUEUED
            )
            running_count = sum(
                1
                for job in active_jobs
                if TrainingJobStatus.is_active(job.status)
                and job.status
                not in [TrainingJobStatus.PENDING, TrainingJobStatus.QUEUED]
            )

            # Calculate average processing time
            avg_processing_time = None
            if self.jobs_processed > 0:
                avg_processing_time = self.total_processing_time / self.jobs_processed

            # Calculate throughput
            uptime_hours = (
                datetime.now(timezone.utc) - self.start_time
            ).total_seconds() / 3600
            throughput = self.jobs_processed / uptime_hours if uptime_hours > 0 else 0

            return {
                "total_jobs": len(active_jobs),
                "pending_jobs": pending_count,
                "queued_jobs": queued_count,
                "running_jobs": running_count,
                "max_concurrent": TrainingConstants.MAX_CONCURRENT_TRAININGS,
                "is_healthy": self.is_healthy,
                "last_processed_at": max(
                    [job.started_at for job in active_jobs if job.started_at],
                    default=None,
                ),
                "available_workers": len(
                    [w for w in self.workers.values() if not w.is_busy]
                ),
                "total_workers": len(self.workers),
                "average_training_time_seconds": avg_processing_time,
                "queue_throughput_per_hour": throughput,
                "jobs_processed_total": self.jobs_processed,
                "jobs_failed_total": self.jobs_failed,
                "uptime_seconds": (
                    datetime.now(timezone.utc) - self.start_time
                ).total_seconds(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get queue info: {e}")
            return {}

    async def get_health_status(self) -> BackgroundTaskHealth:
        """Get detailed health status of the background task system"""
        try:
            # Get system resource usage
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent()
            disk_percent = psutil.disk_usage("/").percent

            # Count active workers
            active_workers = len([w for w in self.workers.values() if w.is_busy])

            # Get recent error count
            current_time = datetime.now(timezone.utc)
            one_hour_ago = current_time - timedelta(hours=1)

            # This would need to be tracked separately for real error counting
            errors_last_hour = 0  # Placeholder

            # Calculate jobs processed in last hour
            # This would need more sophisticated tracking
            jobs_last_hour = 0  # Placeholder

            # Calculate average job duration
            avg_duration = None
            if self.jobs_processed > 0:
                avg_duration = self.total_processing_time / self.jobs_processed

            return BackgroundTaskHealth(
                is_healthy=self.is_healthy,
                active_workers=active_workers,
                total_workers=len(self.workers),
                queue_size=self.job_queue.qsize(),
                max_queue_size=TrainingConstants.MAX_QUEUE_SIZE,
                jobs_processed_last_hour=jobs_last_hour,
                average_job_duration_seconds=avg_duration,
                memory_usage_percent=memory_percent,
                cpu_usage_percent=cpu_percent,
                disk_usage_percent=disk_percent,
                errors_last_hour=errors_last_hour,
                last_error_message=(
                    self.health_issues[-1] if self.health_issues else None
                ),
                last_health_check=self.last_health_check,
            )

        except Exception as e:
            self.logger.error(f"Failed to get health status: {e}")
            return BackgroundTaskHealth(
                is_healthy=False,
                active_workers=0,
                total_workers=0,
                queue_size=0,
                max_queue_size=0,
                last_health_check=datetime.now(timezone.utc),
            )

    # Private methods

    def _initialize_workers(self) -> None:
        """Initialize worker pool"""
        try:
            for i in range(TrainingConstants.MAX_CONCURRENT_TRAININGS):
                worker_id = f"worker_{i+1}"
                self.workers[worker_id] = TaskWorker(worker_id=worker_id)

            self.logger.info(f"Initialized {len(self.workers)} workers")

        except Exception as e:
            self.logger.error(f"Failed to initialize workers: {e}")
            raise

    def _start_background_tasks(self) -> None:
        """Start background management tasks"""
        try:
            # Task manager - processes queue
            self._manager_task = asyncio.create_task(self._task_manager_loop())

            # Health monitor - monitors system health
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

            # Cleanup task - cleans up completed jobs
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self.logger.info("Started background tasks")

        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")
            raise

    async def _task_manager_loop(self) -> None:
        """Main task management loop"""
        while True:
            try:
                # Get next job from priority queue
                if not self.priority_queue.empty():
                    priority_score, task_data = await self.priority_queue.get()

                    job_id = task_data["job_id"]

                    # Check if job was cancelled
                    if job_id in self.cancelled_jobs:
                        self.cancelled_jobs.remove(job_id)
                        continue

                    # Find available worker
                    available_worker = self._get_available_worker()
                    if available_worker:
                        # Assign job to worker
                        available_worker.is_busy = True
                        available_worker.current_job_id = job_id
                        available_worker.started_at = datetime.now(timezone.utc)
                        available_worker.last_heartbeat = datetime.now(timezone.utc)

                        # Create and start task
                        task = asyncio.create_task(
                            self._execute_training_task(task_data, available_worker)
                        )
                        self.active_tasks[job_id] = task

                        self.logger.info(
                            f"Started job {job_id} on {available_worker.worker_id}"
                        )
                    else:
                        # No workers available, put job back in queue
                        await self.priority_queue.put((priority_score, task_data))

                # Sleep briefly to prevent busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in task manager loop: {e}")
                await asyncio.sleep(5)

    async def _execute_training_task(
        self, task_data: Dict[str, Any], worker: TaskWorker
    ) -> None:
        """Execute a training task"""
        job_id = task_data["job_id"]
        request = task_data["request"]
        training_function = task_data["training_function"]
        progress_callback = task_data.get("progress_callback")

        start_time = time.time()

        try:
            self.logger.info(f"Executing training job {job_id}")

            # Update job status
            await self.job_facade.update_job(
                job_id,
                {
                    "status": TrainingJobStatus.INITIALIZING,
                    "started_at": datetime.now(timezone.utc),
                    "worker_id": worker.worker_id,
                    "progress": TrainingConstants.PROGRESS_STAGES[
                        TrainingJobStatus.INITIALIZING
                    ],
                },
            )

            # Create progress callback wrapper
            async def wrapped_progress_callback(
                status: TrainingJobStatus, progress: float, message: str = "", **kwargs
            ):
                try:
                    # Check if job was cancelled
                    if job_id in self.cancelled_jobs:
                        raise asyncio.CancelledError("Job was cancelled")

                    # Update worker heartbeat
                    worker.last_heartbeat = datetime.now(timezone.utc)

                    # Create progress update
                    progress_update = TrainingProgressUpdate(
                        job_id=job_id,
                        status=status,
                        progress=progress,
                        current_stage=message,
                        elapsed_seconds=time.time() - start_time,
                        **kwargs,
                    )

                    # Update repository
                    await self.job_facade.update_progress(job_id, progress_update)

                    # Call user progress callback if provided
                    if progress_callback:
                        await progress_callback(progress_update)

                except Exception as e:
                    self.logger.warning(
                        f"Error in progress callback for job {job_id}: {e}"
                    )

            # Execute training function
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: asyncio.run(
                    training_function(request, wrapped_progress_callback)
                ),
            )

            # Calculate duration
            duration = time.time() - start_time

            # Update job with results
            if result.get("success", False):
                await self.job_facade.update_job(
                    job_id,
                    {
                        "status": TrainingJobStatus.COMPLETED,
                        "progress": 1.0,
                        "completed_at": datetime.now(timezone.utc),
                        "duration_seconds": duration,
                        "training_metrics": result.get("training_metrics", {}),
                        "validation_metrics": result.get("validation_metrics", {}),
                        "model_path": result.get("model_path"),
                        "model_size_bytes": result.get("model_size_bytes"),
                    },
                )

                self.jobs_processed += 1
                self.total_processing_time += duration

                self.logger.info(f"Completed job {job_id} in {duration:.2f}s")
            else:
                # Training failed
                await self.job_facade.update_job(
                    job_id,
                    {
                        "status": TrainingJobStatus.FAILED,
                        "completed_at": datetime.now(timezone.utc),
                        "duration_seconds": duration,
                        "error_message": result.get("error", "Training failed"),
                        "error_code": result.get("error_code", "E003"),
                    },
                )

                self.jobs_failed += 1

                self.logger.error(
                    f"Job {job_id} failed: {result.get('error', 'Unknown error')}"
                )

        except asyncio.CancelledError:
            # Job was cancelled
            await self.job_facade.update_job(
                job_id,
                {
                    "status": TrainingJobStatus.CANCELLED,
                    "completed_at": datetime.now(timezone.utc),
                    "duration_seconds": time.time() - start_time,
                    "error_message": "Job was cancelled",
                },
            )

            self.logger.info(f"Job {job_id} was cancelled")

        except Exception as e:
            # Unexpected error
            await self.job_facade.update_job(
                job_id,
                {
                    "status": TrainingJobStatus.FAILED,
                    "completed_at": datetime.now(timezone.utc),
                    "duration_seconds": time.time() - start_time,
                    "error_message": str(e),
                    "error_code": "E999",
                },
            )

            self.jobs_failed += 1

            self.logger.error(f"Job {job_id} failed with unexpected error: {e}")

        finally:
            # Clean up worker
            worker.is_busy = False
            worker.current_job_id = None
            worker.started_at = None
            worker.processed_jobs += 1

            # Remove from active tasks
            self.active_tasks.pop(job_id, None)

            # Remove from cancelled set if present
            self.cancelled_jobs.discard(job_id)

    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop"""
        while True:
            try:
                await self._check_system_health()
                await asyncio.sleep(BackgroundTaskConfig.HEALTH_CHECK_INTERVAL)

            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(30)

    async def _check_system_health(self) -> None:
        """Check system health and update status"""
        try:
            self.last_health_check = datetime.now(timezone.utc)
            issues = []

            # Check memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > BackgroundTaskConfig.MEMORY_THRESHOLD_MB:
                issues.append(f"High memory usage: {memory_percent:.1f}%")

            # Check CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > BackgroundTaskConfig.CPU_THRESHOLD_PERCENT:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")

            # Check disk usage
            disk_percent = psutil.disk_usage("/").percent
            if disk_percent > BackgroundTaskConfig.DISK_THRESHOLD_PERCENT:
                issues.append(f"High disk usage: {disk_percent:.1f}%")

            # Check worker heartbeats
            current_time = datetime.now(timezone.utc)
            for worker in self.workers.values():
                if worker.is_busy and worker.last_heartbeat:
                    time_since_heartbeat = (
                        current_time - worker.last_heartbeat
                    ).total_seconds()
                    if (
                        time_since_heartbeat
                        > BackgroundTaskConfig.TASK_HEARTBEAT_INTERVAL * 2
                    ):
                        issues.append(f"Worker {worker.worker_id} not responding")

            # Update health status
            self.is_healthy = len(issues) == 0
            self.health_issues = issues

            if issues:
                self.logger.warning(f"System health issues: {', '.join(issues)}")

        except Exception as e:
            self.logger.error(f"Failed to check system health: {e}")
            self.is_healthy = False
            self.health_issues = [f"Health check failed: {str(e)}"]

    async def _cleanup_loop(self) -> None:
        """Cleanup loop for completed tasks"""
        while True:
            try:
                # Clean up completed tasks
                completed_tasks = [
                    job_id for job_id, task in self.active_tasks.items() if task.done()
                ]

                for job_id in completed_tasks:
                    self.active_tasks.pop(job_id, None)

                if completed_tasks:
                    self.logger.debug(
                        f"Cleaned up {len(completed_tasks)} completed tasks"
                    )

                await asyncio.sleep(BackgroundTaskConfig.TASK_CLEANUP_BATCH_SIZE)

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)

    def _get_available_worker(self) -> Optional[TaskWorker]:
        """Get an available worker"""
        for worker in self.workers.values():
            if not worker.is_busy:
                return worker
        return None

    def _calculate_priority_score(self, priority: TrainingJobPriority) -> int:
        """Calculate priority score for queue ordering (lower = higher priority)"""
        priority_scores = {
            TrainingJobPriority.URGENT: 1,
            TrainingJobPriority.HIGH: 2,
            TrainingJobPriority.NORMAL: 3,
            TrainingJobPriority.LOW: 4,
        }
        return priority_scores.get(priority, 3)

    def _estimate_training_duration(self, request: AsyncTrainingRequest) -> int:
        """Estimate training duration based on configuration"""
        base_duration = 300  # 5 minutes base

        # Adjust based on epochs
        epoch_factor = request.config.num_epochs / 10

        # Adjust based on model type
        model_factors = {
            "patchtst": 1.0,
            "patchtsmixer": 1.2,
            "pytorch_lightning_transformer": 1.5,
        }
        model_factor = model_factors.get(request.model_type.value, 1.0)

        # Adjust based on batch size (smaller = longer)
        batch_factor = 64 / request.config.batch_size

        estimated = int(base_duration * epoch_factor * model_factor * batch_factor)
        return max(60, min(estimated, 3600))  # Between 1 minute and 1 hour

    async def shutdown(self) -> None:
        """Shutdown the background task manager"""
        try:
            self.logger.info("Shutting down background task manager...")

            # Cancel all active tasks
            for job_id, task in self.active_tasks.items():
                task.cancel()

            # Wait for tasks to complete
            if self.active_tasks:
                await asyncio.gather(
                    *self.active_tasks.values(), return_exceptions=True
                )

            # Cancel background tasks
            for task in [
                self._manager_task,
                self._health_monitor_task,
                self._cleanup_task,
            ]:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Shutdown executor
            self.executor.shutdown(wait=True)

            self.logger.info("Background task manager shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during background task manager shutdown: {e}")
