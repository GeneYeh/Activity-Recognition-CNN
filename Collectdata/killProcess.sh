var=`ps aux |grep Collectdata.py | awk '{print $2}'`
for data in $var
do
	sudo kill -9 $data
done

