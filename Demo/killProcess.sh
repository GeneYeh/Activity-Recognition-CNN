var=`ps aux |grep Demo.py | awk '{print $2}'`
for data in $var
do
	sudo kill -9 $data
done

