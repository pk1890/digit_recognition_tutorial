Neural network for handwritten digit-recognition made with tutorial on http://neuralnetworksanddeeplearning.com/chap1.html


How to use it:

1. launch python2.7 with numpy, scipy and PIL (if you have Anaconda you have it at default)
2. type import network to load code
3. create new neural network by typing  net = Network.Network([784, 30, 10]) # (argument is table of layers sizes)
4. type net.learn_on_mnist(3.0, 30, 10) to start learning process (first arg is learning rate, second epochs number, third data batch size)
5. create .bmp image 28x28px and draw digit
6. type net.run("relative_path_to_your_file") to let network guess what digit you have drew