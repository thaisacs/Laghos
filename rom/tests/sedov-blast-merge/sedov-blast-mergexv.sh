NUM_PARALLEL_PROCESSORS=2
testNames=(offline romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.05 -offline -ef 0.9999 -writesol -romsvds -romos -rostype load -romsrhs -romxandv
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.05 -online -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -soldiff -romsrhs -romgs -romhr -rdimx 24 -rdimv 60 -rdime 20 -rdimfv 110 -rdimfe 40 -romxandv -efx 0.99999
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -restore -romsrhs -rdimx 24 -rdimv 60 -rdime 20 -romxandv -efx 0.99999
    ;;
esac