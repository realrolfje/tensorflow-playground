# Runs the tfds executable to create tensorflow datasets
#
# Create a new dataset with:
# ./tfds.sh new my_dataset
#
docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow tfds "$@"