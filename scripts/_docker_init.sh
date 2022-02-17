#!/usr/bin/env bash

# overwrite uid and gid
usermod -u $HOST_UID developer
groupmod -g $HOST_GID developer

# keep some environments
echo "export PYTHONPATH=${PYTHONPATH}" >> /home/developer/.bashrc
echo "export TVM_LIBRARY_PATH=${TVM_LIBRARY_PATH}" >> /home/developer/.bashrc
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> /home/developer/.bashrc
echo "export PATH=${PATH}" >> /home/developer/.bashrc
echo "cd /workspaces/arachne" >> /home/developer/.bashrc

# change to the developer
chown developer:developer -R /home/developer
su - developer