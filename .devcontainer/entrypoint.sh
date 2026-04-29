#!/bin/sh

sudo setfacl -R -m u:ubuntu:rwX /workspace
sudo setfacl -R -d -m u:ubuntu:rwX /workspace
sudo setfacl -R -m u:ubuntu:rwX /home/ubuntu/
sudo setfacl -R -d -m u:ubuntu:rwX /home/ubuntu/

# install dependencies
uv sync --frozen --all-extras


# Run the CMD, as the main container process
# exec "$@"
$@
