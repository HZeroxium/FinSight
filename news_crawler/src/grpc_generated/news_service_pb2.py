# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: news_service.proto
# Protobuf Python Version: 6.31.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'news_service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12news_service.proto\x12\x0cnews_crawler\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1bgoogle/protobuf/empty.proto\"\xbd\x01\n\x10NewsItemResponse\x12(\n\x06source\x18\x01 \x01(\x0e\x32\x18.news_crawler.NewsSource\x12\r\n\x05title\x18\x02 \x01(\t\x12\x0b\n\x03url\x18\x03 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x04 \x01(\t\x12\x30\n\x0cpublished_at\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0e\n\x06\x61uthor\x18\x06 \x01(\t\x12\x0c\n\x04tags\x18\x07 \x03(\t\"\xa1\x02\n\x0e\x46iltersApplied\x12\x0e\n\x06source\x18\x01 \x01(\t\x12\x10\n\x08keywords\x18\x02 \x03(\t\x12\x0c\n\x04tags\x18\x03 \x03(\t\x12.\n\nstart_date\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12,\n\x08\x65nd_date\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x17\n\x0fhas_date_filter\x18\x06 \x01(\x08\x12\x1b\n\x13has_keywords_filter\x18\x07 \x01(\x08\x12\x17\n\x0fhas_tags_filter\x18\x08 \x01(\x08\x12\x18\n\x10time_range_hours\x18\t \x01(\x05\x12\x18\n\x10is_recent_search\x18\n \x01(\x08\"\xba\x01\n\x0cNewsResponse\x12-\n\x05items\x18\x01 \x03(\x0b\x32\x1e.news_crawler.NewsItemResponse\x12\x13\n\x0btotal_count\x18\x02 \x01(\x05\x12\r\n\x05limit\x18\x03 \x01(\x05\x12\x0e\n\x06offset\x18\x04 \x01(\x05\x12\x10\n\x08has_more\x18\x05 \x01(\x08\x12\x35\n\x0f\x66ilters_applied\x18\x06 \x01(\x0b\x32\x1c.news_crawler.FiltersApplied\"\xda\x01\n\x11SearchNewsRequest\x12(\n\x06source\x18\x01 \x01(\x0e\x32\x18.news_crawler.NewsSource\x12\x10\n\x08keywords\x18\x02 \x03(\t\x12\x0c\n\x04tags\x18\x03 \x03(\t\x12.\n\nstart_date\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12,\n\x08\x65nd_date\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\r\n\x05limit\x18\x06 \x01(\x05\x12\x0e\n\x06offset\x18\x07 \x01(\x05\"^\n\x14GetRecentNewsRequest\x12\r\n\x05hours\x18\x01 \x01(\x05\x12(\n\x06source\x18\x02 \x01(\x0e\x32\x18.news_crawler.NewsSource\x12\r\n\x05limit\x18\x03 \x01(\x05\"\xbf\x01\n\x16GetNewsBySourceRequest\x12(\n\x06source\x18\x01 \x01(\x0e\x32\x18.news_crawler.NewsSource\x12\r\n\x05limit\x18\x02 \x01(\x05\x12\x0e\n\x06offset\x18\x03 \x01(\x05\x12.\n\nstart_date\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12,\n\x08\x65nd_date\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"s\n\x17SearchByKeywordsRequest\x12\x10\n\x08keywords\x18\x01 \x03(\t\x12(\n\x06source\x18\x02 \x01(\x0e\x32\x18.news_crawler.NewsSource\x12\r\n\x05limit\x18\x03 \x01(\x05\x12\r\n\x05hours\x18\x04 \x01(\x05\"|\n\x14GetNewsByTagsRequest\x12\x0c\n\x04tags\x18\x01 \x03(\t\x12(\n\x06source\x18\x02 \x01(\x0e\x32\x18.news_crawler.NewsSource\x12\r\n\x05limit\x18\x03 \x01(\x05\x12\x0e\n\x06offset\x18\x04 \x01(\x05\x12\r\n\x05hours\x18\x05 \x01(\x05\"%\n\x12GetNewsItemRequest\x12\x0f\n\x07item_id\x18\x01 \x01(\t\"R\n\x17GetAvailableTagsRequest\x12(\n\x06source\x18\x01 \x01(\x0e\x32\x18.news_crawler.NewsSource\x12\r\n\x05limit\x18\x02 \x01(\x05\"`\n\x15\x41vailableTagsResponse\x12\x0c\n\x04tags\x18\x01 \x03(\t\x12\x13\n\x0btotal_count\x18\x02 \x01(\x05\x12\x15\n\rsource_filter\x18\x03 \x01(\t\x12\r\n\x05limit\x18\x04 \x01(\x05\"\xbb\x03\n\x11NewsStatsResponse\x12\x16\n\x0etotal_articles\x18\x01 \x01(\x05\x12Q\n\x12\x61rticles_by_source\x18\x02 \x03(\x0b\x32\x35.news_crawler.NewsStatsResponse.ArticlesBySourceEntry\x12\x1b\n\x13recent_articles_24h\x18\x03 \x01(\x05\x12\x32\n\x0eoldest_article\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x32\n\x0enewest_article\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12H\n\rdatabase_info\x18\x06 \x03(\x0b\x32\x31.news_crawler.NewsStatsResponse.DatabaseInfoEntry\x1a\x37\n\x15\x41rticlesBySourceEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x33\n\x11\x44\x61tabaseInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"(\n\x15\x44\x65leteNewsItemRequest\x12\x0f\n\x07item_id\x18\x01 \x01(\t\"2\n\x0e\x44\x65leteResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t*b\n\nNewsSource\x12\x1b\n\x17NEWS_SOURCE_UNSPECIFIED\x10\x00\x12\x18\n\x14NEWS_SOURCE_COINDESK\x10\x01\x12\x1d\n\x19NEWS_SOURCE_COINTELEGRAPH\x10\x02\x32\xfa\x05\n\x0bNewsService\x12I\n\nSearchNews\x12\x1f.news_crawler.SearchNewsRequest\x1a\x1a.news_crawler.NewsResponse\x12O\n\rGetRecentNews\x12\".news_crawler.GetRecentNewsRequest\x1a\x1a.news_crawler.NewsResponse\x12S\n\x0fGetNewsBySource\x12$.news_crawler.GetNewsBySourceRequest\x1a\x1a.news_crawler.NewsResponse\x12U\n\x10SearchByKeywords\x12%.news_crawler.SearchByKeywordsRequest\x1a\x1a.news_crawler.NewsResponse\x12O\n\rGetNewsByTags\x12\".news_crawler.GetNewsByTagsRequest\x1a\x1a.news_crawler.NewsResponse\x12O\n\x0bGetNewsItem\x12 .news_crawler.GetNewsItemRequest\x1a\x1e.news_crawler.NewsItemResponse\x12^\n\x10GetAvailableTags\x12%.news_crawler.GetAvailableTagsRequest\x1a#.news_crawler.AvailableTagsResponse\x12L\n\x11GetNewsStatistics\x12\x16.google.protobuf.Empty\x1a\x1f.news_crawler.NewsStatsResponse\x12S\n\x0e\x44\x65leteNewsItem\x12#.news_crawler.DeleteNewsItemRequest\x1a\x1c.news_crawler.DeleteResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'news_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_NEWSSTATSRESPONSE_ARTICLESBYSOURCEENTRY']._loaded_options = None
  _globals['_NEWSSTATSRESPONSE_ARTICLESBYSOURCEENTRY']._serialized_options = b'8\001'
  _globals['_NEWSSTATSRESPONSE_DATABASEINFOENTRY']._loaded_options = None
  _globals['_NEWSSTATSRESPONSE_DATABASEINFOENTRY']._serialized_options = b'8\001'
  _globals['_NEWSSOURCE']._serialized_start=2286
  _globals['_NEWSSOURCE']._serialized_end=2384
  _globals['_NEWSITEMRESPONSE']._serialized_start=99
  _globals['_NEWSITEMRESPONSE']._serialized_end=288
  _globals['_FILTERSAPPLIED']._serialized_start=291
  _globals['_FILTERSAPPLIED']._serialized_end=580
  _globals['_NEWSRESPONSE']._serialized_start=583
  _globals['_NEWSRESPONSE']._serialized_end=769
  _globals['_SEARCHNEWSREQUEST']._serialized_start=772
  _globals['_SEARCHNEWSREQUEST']._serialized_end=990
  _globals['_GETRECENTNEWSREQUEST']._serialized_start=992
  _globals['_GETRECENTNEWSREQUEST']._serialized_end=1086
  _globals['_GETNEWSBYSOURCEREQUEST']._serialized_start=1089
  _globals['_GETNEWSBYSOURCEREQUEST']._serialized_end=1280
  _globals['_SEARCHBYKEYWORDSREQUEST']._serialized_start=1282
  _globals['_SEARCHBYKEYWORDSREQUEST']._serialized_end=1397
  _globals['_GETNEWSBYTAGSREQUEST']._serialized_start=1399
  _globals['_GETNEWSBYTAGSREQUEST']._serialized_end=1523
  _globals['_GETNEWSITEMREQUEST']._serialized_start=1525
  _globals['_GETNEWSITEMREQUEST']._serialized_end=1562
  _globals['_GETAVAILABLETAGSREQUEST']._serialized_start=1564
  _globals['_GETAVAILABLETAGSREQUEST']._serialized_end=1646
  _globals['_AVAILABLETAGSRESPONSE']._serialized_start=1648
  _globals['_AVAILABLETAGSRESPONSE']._serialized_end=1744
  _globals['_NEWSSTATSRESPONSE']._serialized_start=1747
  _globals['_NEWSSTATSRESPONSE']._serialized_end=2190
  _globals['_NEWSSTATSRESPONSE_ARTICLESBYSOURCEENTRY']._serialized_start=2082
  _globals['_NEWSSTATSRESPONSE_ARTICLESBYSOURCEENTRY']._serialized_end=2137
  _globals['_NEWSSTATSRESPONSE_DATABASEINFOENTRY']._serialized_start=2139
  _globals['_NEWSSTATSRESPONSE_DATABASEINFOENTRY']._serialized_end=2190
  _globals['_DELETENEWSITEMREQUEST']._serialized_start=2192
  _globals['_DELETENEWSITEMREQUEST']._serialized_end=2232
  _globals['_DELETERESPONSE']._serialized_start=2234
  _globals['_DELETERESPONSE']._serialized_end=2284
  _globals['_NEWSSERVICE']._serialized_start=2387
  _globals['_NEWSSERVICE']._serialized_end=3149
# @@protoc_insertion_point(module_scope)
