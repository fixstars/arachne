# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: fileserver.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import stream_data_pb2 as stream__data__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='fileserver.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10\x66ileserver.proto\x1a\x11stream_data.proto\"D\n\rUploadRequest\x12\x17\n\x05\x63hunk\x18\x01 \x01(\x0b\x32\x06.ChunkH\x00\x12\x12\n\x08\x66ilename\x18\x02 \x01(\tH\x00\x42\x06\n\x04info\"\"\n\x0eUploadResponse\x12\x10\n\x08\x66ilepath\x18\x01 \x01(\t\"\x13\n\x11MakeTmpDirRequest\"%\n\x12MakeTmpDirResponse\x12\x0f\n\x07\x64irname\x18\x01 \x01(\t\"&\n\x13\x44\x65leteTmpDirRequest\x12\x0f\n\x07\x64irname\x18\x01 \x01(\t\"\x16\n\x14\x44\x65leteTmpDirResponse2\xb5\x01\n\nFileServer\x12\x38\n\x0bmake_tmpdir\x12\x12.MakeTmpDirRequest\x1a\x13.MakeTmpDirResponse\"\x00\x12>\n\rdelete_tmpdir\x12\x14.DeleteTmpDirRequest\x1a\x15.DeleteTmpDirResponse\"\x00\x12-\n\x06upload\x12\x0e.UploadRequest\x1a\x0f.UploadResponse\"\x00(\x01\x62\x06proto3'
  ,
  dependencies=[stream__data__pb2.DESCRIPTOR,])




_UPLOADREQUEST = _descriptor.Descriptor(
  name='UploadRequest',
  full_name='UploadRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='chunk', full_name='UploadRequest.chunk', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='filename', full_name='UploadRequest.filename', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='info', full_name='UploadRequest.info',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=39,
  serialized_end=107,
)


_UPLOADRESPONSE = _descriptor.Descriptor(
  name='UploadResponse',
  full_name='UploadResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='filepath', full_name='UploadResponse.filepath', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=109,
  serialized_end=143,
)


_MAKETMPDIRREQUEST = _descriptor.Descriptor(
  name='MakeTmpDirRequest',
  full_name='MakeTmpDirRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=145,
  serialized_end=164,
)


_MAKETMPDIRRESPONSE = _descriptor.Descriptor(
  name='MakeTmpDirResponse',
  full_name='MakeTmpDirResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dirname', full_name='MakeTmpDirResponse.dirname', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=166,
  serialized_end=203,
)


_DELETETMPDIRREQUEST = _descriptor.Descriptor(
  name='DeleteTmpDirRequest',
  full_name='DeleteTmpDirRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dirname', full_name='DeleteTmpDirRequest.dirname', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=205,
  serialized_end=243,
)


_DELETETMPDIRRESPONSE = _descriptor.Descriptor(
  name='DeleteTmpDirResponse',
  full_name='DeleteTmpDirResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=245,
  serialized_end=267,
)

_UPLOADREQUEST.fields_by_name['chunk'].message_type = stream__data__pb2._CHUNK
_UPLOADREQUEST.oneofs_by_name['info'].fields.append(
  _UPLOADREQUEST.fields_by_name['chunk'])
_UPLOADREQUEST.fields_by_name['chunk'].containing_oneof = _UPLOADREQUEST.oneofs_by_name['info']
_UPLOADREQUEST.oneofs_by_name['info'].fields.append(
  _UPLOADREQUEST.fields_by_name['filename'])
_UPLOADREQUEST.fields_by_name['filename'].containing_oneof = _UPLOADREQUEST.oneofs_by_name['info']
DESCRIPTOR.message_types_by_name['UploadRequest'] = _UPLOADREQUEST
DESCRIPTOR.message_types_by_name['UploadResponse'] = _UPLOADRESPONSE
DESCRIPTOR.message_types_by_name['MakeTmpDirRequest'] = _MAKETMPDIRREQUEST
DESCRIPTOR.message_types_by_name['MakeTmpDirResponse'] = _MAKETMPDIRRESPONSE
DESCRIPTOR.message_types_by_name['DeleteTmpDirRequest'] = _DELETETMPDIRREQUEST
DESCRIPTOR.message_types_by_name['DeleteTmpDirResponse'] = _DELETETMPDIRRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

UploadRequest = _reflection.GeneratedProtocolMessageType('UploadRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADREQUEST,
  '__module__' : 'fileserver_pb2'
  # @@protoc_insertion_point(class_scope:UploadRequest)
  })
_sym_db.RegisterMessage(UploadRequest)

UploadResponse = _reflection.GeneratedProtocolMessageType('UploadResponse', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADRESPONSE,
  '__module__' : 'fileserver_pb2'
  # @@protoc_insertion_point(class_scope:UploadResponse)
  })
_sym_db.RegisterMessage(UploadResponse)

MakeTmpDirRequest = _reflection.GeneratedProtocolMessageType('MakeTmpDirRequest', (_message.Message,), {
  'DESCRIPTOR' : _MAKETMPDIRREQUEST,
  '__module__' : 'fileserver_pb2'
  # @@protoc_insertion_point(class_scope:MakeTmpDirRequest)
  })
_sym_db.RegisterMessage(MakeTmpDirRequest)

MakeTmpDirResponse = _reflection.GeneratedProtocolMessageType('MakeTmpDirResponse', (_message.Message,), {
  'DESCRIPTOR' : _MAKETMPDIRRESPONSE,
  '__module__' : 'fileserver_pb2'
  # @@protoc_insertion_point(class_scope:MakeTmpDirResponse)
  })
_sym_db.RegisterMessage(MakeTmpDirResponse)

DeleteTmpDirRequest = _reflection.GeneratedProtocolMessageType('DeleteTmpDirRequest', (_message.Message,), {
  'DESCRIPTOR' : _DELETETMPDIRREQUEST,
  '__module__' : 'fileserver_pb2'
  # @@protoc_insertion_point(class_scope:DeleteTmpDirRequest)
  })
_sym_db.RegisterMessage(DeleteTmpDirRequest)

DeleteTmpDirResponse = _reflection.GeneratedProtocolMessageType('DeleteTmpDirResponse', (_message.Message,), {
  'DESCRIPTOR' : _DELETETMPDIRRESPONSE,
  '__module__' : 'fileserver_pb2'
  # @@protoc_insertion_point(class_scope:DeleteTmpDirResponse)
  })
_sym_db.RegisterMessage(DeleteTmpDirResponse)



_FILESERVER = _descriptor.ServiceDescriptor(
  name='FileServer',
  full_name='FileServer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=270,
  serialized_end=451,
  methods=[
  _descriptor.MethodDescriptor(
    name='make_tmpdir',
    full_name='FileServer.make_tmpdir',
    index=0,
    containing_service=None,
    input_type=_MAKETMPDIRREQUEST,
    output_type=_MAKETMPDIRRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='delete_tmpdir',
    full_name='FileServer.delete_tmpdir',
    index=1,
    containing_service=None,
    input_type=_DELETETMPDIRREQUEST,
    output_type=_DELETETMPDIRRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='upload',
    full_name='FileServer.upload',
    index=2,
    containing_service=None,
    input_type=_UPLOADREQUEST,
    output_type=_UPLOADRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_FILESERVER)

DESCRIPTOR.services_by_name['FileServer'] = _FILESERVER

# @@protoc_insertion_point(module_scope)
