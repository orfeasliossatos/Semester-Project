clear
sbatch $1
while :
do
	echo "Monitoring squeue and latest job"
	squeue 
	nb=-1
	fn=""
	dr="/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs"
	for file in "$dr"/*; do
		if [ -f "$file" ]; then
			number=$(basename "$file" | grep -oE '[0-9]+')
			if [ "$number" -gt "$nb" ]; then
				nb="$number"
				fn="$file"
			fi
		fi
	done

	echo "Latest job output $nb"
	echo "Error files :"
	cat $dr/$nb.err
	echo "-------------------"
	echo "Output files"
	cat $dr/$nb.out
	sleep 5

	clear
done
