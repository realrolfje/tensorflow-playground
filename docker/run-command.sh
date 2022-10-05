# Runs the python script given as name on the command line
docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow "$@"