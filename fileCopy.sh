
echo "target" $1
echo "destination" $2

sshpass -f "pipass.txt" scp -P 3113 $1 pi@144.173.65.145:/home/pi/$2
echo "1/5"
sshpass -f "pipass.txt" scp -P 3112 $1 pi@144.173.65.145:/home/pi/$2
echo "2/5"
sshpass -f "pipass.txt" scp -P 3105 $1 pi@144.173.65.145:/home/pi/$2
echo "3/5"
sshpass -f "pipass.txt" scp -P 3106 $1 pi@144.173.65.145:/home/pi/$2
echo "4/5"
sshpass -f "pipass.txt" scp -P 3107 $1 jz454@144.173.65.145:/home/jz454/$2
echo "5/5"
