# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import forecast_server_pb2 as forecast__server__pb2

GRPC_GENERATED_VERSION = '1.70.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in forecast_server_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class ForecastServiceStub(object):
    """Service definition
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CreateForecast = channel.unary_unary(
                '/forecast.ForecastService/CreateForecast',
                request_serializer=forecast__server__pb2.ForecastRequest.SerializeToString,
                response_deserializer=forecast__server__pb2.ForecastResponse.FromString,
                _registered_method=True)
        self.GetDefaultParameters = channel.unary_unary(
                '/forecast.ForecastService/GetDefaultParameters',
                request_serializer=forecast__server__pb2.Empty.SerializeToString,
                response_deserializer=forecast__server__pb2.ProphetParameters.FromString,
                _registered_method=True)


class ForecastServiceServicer(object):
    """Service definition
    """

    def CreateForecast(self, request, context):
        """Maps to POST /forecast/
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDefaultParameters(self, request, context):
        """Maps to GET /parameters/default
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ForecastServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'CreateForecast': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateForecast,
                    request_deserializer=forecast__server__pb2.ForecastRequest.FromString,
                    response_serializer=forecast__server__pb2.ForecastResponse.SerializeToString,
            ),
            'GetDefaultParameters': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDefaultParameters,
                    request_deserializer=forecast__server__pb2.Empty.FromString,
                    response_serializer=forecast__server__pb2.ProphetParameters.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'forecast.ForecastService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('forecast.ForecastService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class ForecastService(object):
    """Service definition
    """

    @staticmethod
    def CreateForecast(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/forecast.ForecastService/CreateForecast',
            forecast__server__pb2.ForecastRequest.SerializeToString,
            forecast__server__pb2.ForecastResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetDefaultParameters(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/forecast.ForecastService/GetDefaultParameters',
            forecast__server__pb2.Empty.SerializeToString,
            forecast__server__pb2.ProphetParameters.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
