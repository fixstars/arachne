DIR="$( cd "$( dirname "$0" )" && pwd -P )"
CODEGEN_DIR=${DIR}/../
echo $DIR
echo $CODEGEN_DIR
python3 -m grpc_tools.protoc -I${DIR} --python_out=${CODEGEN_DIR} --grpc_python_out=${CODEGEN_DIR} ${DIR}/*.proto
