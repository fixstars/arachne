#!/usr/bin/env bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# This script is a wrapper creating the same user inside container as the one
# running the docker/build.sh outside the container. It also set the home directory
# for the user inside container to match the same absolute path as the workspace
# outside of container.  Do not run this manually. It does not make sense. It is
# intended to be called by ci_build.sh only.

set -e

COMMAND=("$@")

if ! touch /this_is_writable_file_system; then
  echo "You can't write to your filesystem!"
  echo "If you are in Docker you should check you do not have too many images" \
      "with too many files in them. Docker has some issue with it."
  exit 1
else
  rm /this_is_writable_file_system
fi

# Added the host user
getent group "${HOST_GID}" || addgroup --gid "${HOST_GID}" "${HOST_GROUP}" --force-badname
getent passwd "${HOST_UID}" || adduser --gid "${HOST_GID}" --uid "${HOST_UID}" \
    --gecos "${USER} (generated by with_the_same_user script)" \
    --disabled-password  --quiet "${HOST_USER}" --force-badname
usermod -a -G sudo "${HOST_USER}"
touch /etc/sudoers.d/90-nopasswd-sudo 
echo "${HOST_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-nopasswd-sudo

# Run
sudo -u "#${HOST_UID}" \
  --preserve-env HOME=/home/${HOST_USER} \
  ${COMMAND[@]}
