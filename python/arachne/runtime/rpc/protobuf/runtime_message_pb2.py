# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: runtime_message.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import msg_response_pb2 as msg__response__pb2
import stream_data_pb2 as stream__data__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='runtime_message.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x15runtime_message.proto\x1a\x12msg_response.proto\x1a\x11stream_data.proto\"!\n\x0bInitRequest\x12\x12\n\nmodel_path\x18\x01 \x01(\t\"J\n\x0fSetInputRequest\x12\x0f\n\x05index\x18\x01 \x01(\x05H\x00\x12\x1e\n\x0cnp_arr_chunk\x18\x02 \x01(\x0b\x32\x06.ChunkH\x00\x42\x06\n\x04\x64\x61ta\"\x0c\n\nRunRequest\"B\n\x10\x42\x65nchmarkRequest\x12\x0e\n\x06warmup\x18\x01 \x01(\x05\x12\x0e\n\x06repeat\x18\x02 \x01(\x05\x12\x0e\n\x06number\x18\x03 \x01(\x05\"T\n\x11\x42\x65nchmarkResponse\x12\x0f\n\x07mean_ts\x18\x01 \x01(\x02\x12\x0e\n\x06std_ts\x18\x02 \x01(\x02\x12\x0e\n\x06max_ts\x18\x03 \x01(\x02\x12\x0e\n\x06min_ts\x18\x04 \x01(\x02\"!\n\x10GetOutputRequest\x12\r\n\x05index\x18\x01 \x01(\x05\"$\n\x11GetOutputResponse\x12\x0f\n\x07np_data\x18\x01 \x01(\x0c\x32\xf1\x01\n\x07Runtime\x12$\n\x04Init\x12\x0c.InitRequest\x1a\x0c.MsgResponse\"\x00\x12.\n\x08SetInput\x12\x10.SetInputRequest\x1a\x0c.MsgResponse\"\x00(\x01\x12\"\n\x03Run\x12\x0b.RunRequest\x1a\x0c.MsgResponse\"\x00\x12\x34\n\tBenchmark\x12\x11.BenchmarkRequest\x1a\x12.BenchmarkResponse\"\x00\x12\x36\n\tGetOutput\x12\x11.GetOutputRequest\x1a\x12.GetOutputResponse\"\x00\x30\x01\x62\x06proto3'
  ,
  dependencies=[msg__response__pb2.DESCRIPTOR,stream__data__pb2.DESCRIPTOR,])




_INITREQUEST = _descriptor.Descriptor(
  name='InitRequest',
  full_name='InitRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_path', full_name='InitRequest.model_path', index=0,
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
  serialized_start=64,
  serialized_end=97,
)


_SETINPUTREQUEST = _descriptor.Descriptor(
  name='SetInputRequest',
  full_name='SetInputRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='index', full_name='SetInputRequest.index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='np_arr_chunk', full_name='SetInputRequest.np_arr_chunk', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
      name='data', full_name='SetInputRequest.data',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=99,
  serialized_end=173,
)


_RUNREQUEST = _descriptor.Descriptor(
  name='RunRequest',
  full_name='RunRequest',
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
  serialized_start=175,
  serialized_end=187,
)


_BENCHMARKREQUEST = _descriptor.Descriptor(
  name='BenchmarkRequest',
  full_name='BenchmarkRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='warmup', full_name='BenchmarkRequest.warmup', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='repeat', full_name='BenchmarkRequest.repeat', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='number', full_name='BenchmarkRequest.number', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=189,
  serialized_end=255,
)


_BENCHMARKRESPONSE = _descriptor.Descriptor(
  name='BenchmarkResponse',
  full_name='BenchmarkResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='mean_ts', full_name='BenchmarkResponse.mean_ts', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='std_ts', full_name='BenchmarkResponse.std_ts', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='max_ts', full_name='BenchmarkResponse.max_ts', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='min_ts', full_name='BenchmarkResponse.min_ts', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=257,
  serialized_end=341,
)


_GETOUTPUTREQUEST = _descriptor.Descriptor(
  name='GetOutputRequest',
  full_name='GetOutputRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='index', full_name='GetOutputRequest.index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=343,
  serialized_end=376,
)


_GETOUTPUTRESPONSE = _descriptor.Descriptor(
  name='GetOutputResponse',
  full_name='GetOutputResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='np_data', full_name='GetOutputResponse.np_data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
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
  serialized_start=378,
  serialized_end=414,
)

_SETINPUTREQUEST.fields_by_name['np_arr_chunk'].message_type = stream__data__pb2._CHUNK
_SETINPUTREQUEST.oneofs_by_name['data'].fields.append(
  _SETINPUTREQUEST.fields_by_name['index'])
_SETINPUTREQUEST.fields_by_name['index'].containing_oneof = _SETINPUTREQUEST.oneofs_by_name['data']
_SETINPUTREQUEST.oneofs_by_name['data'].fields.append(
  _SETINPUTREQUEST.fields_by_name['np_arr_chunk'])
_SETINPUTREQUEST.fields_by_name['np_arr_chunk'].containing_oneof = _SETINPUTREQUEST.oneofs_by_name['data']
DESCRIPTOR.message_types_by_name['InitRequest'] = _INITREQUEST
DESCRIPTOR.message_types_by_name['SetInputRequest'] = _SETINPUTREQUEST
DESCRIPTOR.message_types_by_name['RunRequest'] = _RUNREQUEST
DESCRIPTOR.message_types_by_name['BenchmarkRequest'] = _BENCHMARKREQUEST
DESCRIPTOR.message_types_by_name['BenchmarkResponse'] = _BENCHMARKRESPONSE
DESCRIPTOR.message_types_by_name['GetOutputRequest'] = _GETOUTPUTREQUEST
DESCRIPTOR.message_types_by_name['GetOutputResponse'] = _GETOUTPUTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InitRequest = _reflection.GeneratedProtocolMessageType('InitRequest', (_message.Message,), {
  'DESCRIPTOR' : _INITREQUEST,
  '__module__' : 'runtime_message_pb2'
  # @@protoc_insertion_point(class_scope:InitRequest)
  })
_sym_db.RegisterMessage(InitRequest)

SetInputRequest = _reflection.GeneratedProtocolMessageType('SetInputRequest', (_message.Message,), {
  'DESCRIPTOR' : _SETINPUTREQUEST,
  '__module__' : 'runtime_message_pb2'
  # @@protoc_insertion_point(class_scope:SetInputRequest)
  })
_sym_db.RegisterMessage(SetInputRequest)

RunRequest = _reflection.GeneratedProtocolMessageType('RunRequest', (_message.Message,), {
  'DESCRIPTOR' : _RUNREQUEST,
  '__module__' : 'runtime_message_pb2'
  # @@protoc_insertion_point(class_scope:RunRequest)
  })
_sym_db.RegisterMessage(RunRequest)

BenchmarkRequest = _reflection.GeneratedProtocolMessageType('BenchmarkRequest', (_message.Message,), {
  'DESCRIPTOR' : _BENCHMARKREQUEST,
  '__module__' : 'runtime_message_pb2'
  # @@protoc_insertion_point(class_scope:BenchmarkRequest)
  })
_sym_db.RegisterMessage(BenchmarkRequest)

BenchmarkResponse = _reflection.GeneratedProtocolMessageType('BenchmarkResponse', (_message.Message,), {
  'DESCRIPTOR' : _BENCHMARKRESPONSE,
  '__module__' : 'runtime_message_pb2'
  # @@protoc_insertion_point(class_scope:BenchmarkResponse)
  })
_sym_db.RegisterMessage(BenchmarkResponse)

GetOutputRequest = _reflection.GeneratedProtocolMessageType('GetOutputRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETOUTPUTREQUEST,
  '__module__' : 'runtime_message_pb2'
  # @@protoc_insertion_point(class_scope:GetOutputRequest)
  })
_sym_db.RegisterMessage(GetOutputRequest)

GetOutputResponse = _reflection.GeneratedProtocolMessageType('GetOutputResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETOUTPUTRESPONSE,
  '__module__' : 'runtime_message_pb2'
  # @@protoc_insertion_point(class_scope:GetOutputResponse)
  })
_sym_db.RegisterMessage(GetOutputResponse)



_RUNTIME = _descriptor.ServiceDescriptor(
  name='Runtime',
  full_name='Runtime',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=417,
  serialized_end=658,
  methods=[
  _descriptor.MethodDescriptor(
    name='Init',
    full_name='Runtime.Init',
    index=0,
    containing_service=None,
    input_type=_INITREQUEST,
    output_type=msg__response__pb2._MSGRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetInput',
    full_name='Runtime.SetInput',
    index=1,
    containing_service=None,
    input_type=_SETINPUTREQUEST,
    output_type=msg__response__pb2._MSGRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Run',
    full_name='Runtime.Run',
    index=2,
    containing_service=None,
    input_type=_RUNREQUEST,
    output_type=msg__response__pb2._MSGRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Benchmark',
    full_name='Runtime.Benchmark',
    index=3,
    containing_service=None,
    input_type=_BENCHMARKREQUEST,
    output_type=_BENCHMARKRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetOutput',
    full_name='Runtime.GetOutput',
    index=4,
    containing_service=None,
    input_type=_GETOUTPUTREQUEST,
    output_type=_GETOUTPUTRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_RUNTIME)

DESCRIPTOR.services_by_name['Runtime'] = _RUNTIME

# @@protoc_insertion_point(module_scope)
