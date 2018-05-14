# CNN---Neural-Style-Transfer
Neural Style Transfer (NST) is one of the most fun techniques in deep learning. As seen below, it merges two images, namely, a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S.

In this example, you are going to generate an image of the Louvre museum in Paris (content image C), mixed with a painting by Claude Monet, a leader of the impressionist movement (style image S).

(refer images)
(see pyfile)

2 - Transfer Learning
Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning.

Following the original NST paper (https://arxiv.org/abs/1508.06576), we will use the VGG network. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the earlier layers) and high level features (at the deeper layers).

Run the following code to load parameters from the VGG model. This may take a few seconds.

(see pyfile)

The model is stored in a python dictionary where each variable name is the key and the corresponding value is a tensor containing that variable's value. To run an image through this network, you just have to feed the image to the model. In TensorFlow, you can do so using the tf.assign function. In particular, you will use the assign function like this:

model["input"].assign(image)
This assigns the image as an input to the model. After this, if you want to access the activations of a particular layer, say layer 4_2 when the network is run on this image, you would run a TensorFlow session on the correct tensor conv4_2, as follows:

sess.run(model["conv4_2"])

3 - Neural Style Transfer
We will build the NST algorithm in three steps:

Build the content cost function  Jcontent(C,G)Jcontent(C,G) 
Build the style cost function  Jstyle(S,G)Jstyle(S,G) 
Put it together to get  J(G)=αJcontent(C,G)+βJstyle(S,G)J(G)=αJcontent(C,G)+βJstyle(S,G) .
3.1 - Computing the content cost
In our running example, the content image C will be the picture of the Louvre Museum in Paris. Run the code below to see a picture of the Louvre.

(see pyfile)

The content image (C) shows the Louvre museum's pyramid surrounded by old Paris buildings, against a sunny sky with a few clouds.

3.1.1 - How do you ensure the generated image G matches the content of the image C?

As we saw in lecture, the earlier (shallower) layers of a ConvNet tend to detect lower-level features such as edges and simple textures, and the later (deeper) layers tend to detect higher-level features such as more complex textures as well as object classes.

We would like the "generated" image G to have similar content as the input image C. Suppose you have chosen some layer's activations to represent the content of an image. In practice, you'll get the most visually pleasing results if you choose a layer in the middle of the network--neither too shallow nor too deep. (After you have finished this exercise, feel free to come back and experiment with using different layers, to see how the results vary.)

So, suppose you have picked one particular hidden layer to use. Now, set the image C as the input to the pretrained VGG network, and run forward propagation. Let  a(C)  be the hidden layer activations in the layer you had chosen. (In lecture, we had written this as  a[l](C) , but here we'll drop the superscript [l]  to simplify the notation.) This will be a  nH×nW×nC  tensor. Repeat this process with the image G: Set G as the input, and run forward progation. Let
a(G) be the corresponding hidden layer activation. We will define as the content cost function as:

Jcontent(C,G)=1/4×nH×nW×nC∑all entries(a(C)−a(G))2
 
Here,  nH,nW  and  nC  are the height, width and number of channels of the hidden layer you have chosen, and appear in a normalization term in the cost. For clarity, note that  a(C)  and  a(G)  are the volumes corresponding to a hidden layer's activations. In order to compute the cost  Jcontent(C,G) , it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below. (Technically this unrolling step isn't needed to compute  Jcontent , but it will be good practice for when you do need to carry out a similar operation later for computing the style const  Jstyle .)

(refer images)

Exercise: Compute the "content cost" using TensorFlow.

Instructions: The 3 steps to implement this function are:

Retrieve dimensions from a_G:
Unroll a_C and a_G as explained in the picture above
Compute the content cost:

(see pyfile)

Expected Output:

J_content	6.76559

What you should remember:

The content cost takes a hidden layer activation of the neural network, and measures how different a(C) and a(G) are.
When we minimize the content cost later, this will help make sure GG has similar content as C.

(see pyfile)

Lets see how you can now define a "style" const function  Jstyle(S,G)Jstyle(S,G) .

3.2.1 - Style matrix
The style matrix is also called a "Gram matrix." In linear algebra, the Gram matrix G of a set of vectors (v1,…,vn) is the matrix of dot products, whose entries are Gij=viTvj=np.dot(vi,vj). In other words, Gij compares how similar vi is to vj: If they are highly similar, you would expect them to have a large dot product, and thus for Gij to be large.

Note that there is an unfortunate collision in the variable names used here. We are following common terminology used in the literature, but G is used to denote the Style matrix (or Gram matrix) as well as to denote the generated image G. We will try to make sure which G we are referring to is always clear from the context.

In NST, you can compute the Style matrix by multiplying the "unrolled" filter matrix with their transpose:

(refer images)

The result is a matrix of dimension  (nC,nC)  where  nC  is the number of filters. The value  Gij  measures how similar the activations of filter  i  are to the activations of filter  j .

One important part of the gram matrix is that the diagonal elements such as  GiiGii  also measures how active filter  i  is. For example, suppose filter  i  is detecting vertical textures in the image. Then  GiiGii  measures how common vertical textures are in the image as a whole: If  Gii  is large, this means that the image has a lot of vertical texture.

By capturing the prevalence of different types of features ( Gii ), as well as how much different features occur together ( Gij ), the Style matrix  G  measures the style of an image.

Exercise: Using TensorFlow, implement a function that computes the Gram matrix of a matrix A. The formula is: The gram matrix of A is  GA=AATGA=AAT .

(see pyfile)

Expected Output:

GA	[[ 6.42230511 -4.42912197 -2.09668207] 
[ -4.42912197 19.46583748 19.56387138] 
[ -2.09668207 19.56387138 20.6864624 ]]

3.2.2 - Style cost
After generating the Style matrix (Gram matrix), your goal will be to minimize the distance between the Gram matrix of the "style" image S and that of the "generated" image G. For now, we are using only a single hidden layer a[l], and the corresponding style cost for this layer is defined as:

J[l]style(S,G)=1/4×nC2×(nH×nW)2∑i=1nC∑j=1nC(G(S)ij−G(G)ij)2
where G(S) and G(G) are respectively the Gram matrices of the "style" image and the "generated" image, computed using the hidden layer activations for a particular hidden layer in the network.

Exercise: Compute the style cost for a single layer.

Instructions: The 3 steps to implement this function are:

Retrieve dimensions from the hidden layer activations a_G:
  To retrieve dimensions from a tensor X, use: X.get_shape().as_list()
Unroll the hidden layer activations a_S and a_G into 2D matrices, as explained in the picture above.
Compute the Style matrix of the images S and G. (Use the function you had previously written.)
Compute the Style cost:

(refer pyfile)

Expected Output:

J_style_layer	9.19028

3.2.3 Style Weights
So far you have captured the style from only one layer. We'll get better results if we "merge" style costs from several different layers. After completing this exercise, feel free to come back and experiment with different weights to see how it changes the generated image GG. But for now, this is a pretty reasonable default:

(see pyfile)

You can combine the style costs for different layers as follows:

Jstyle(S,G)=∑lλ[l]J[l]style(S,G)
 
where the values for λ[l]  are given in STYLE_LAYERS.

We've implemented a compute_style_cost(...) function. It simply calls your compute_layer_style_cost(...) several times, and weights their results using the values in STYLE_LAYERS. Read over it to make sure you understand what it's doing.

(see pyfile)

Note: In the inner-loop of the for-loop above, a_G is a tensor and hasn't been evaluated yet. It will be evaluated and updated at each iteration when we run the TensorFlow graph in model_nn() below.

What you should remember:

The style of an image can be represented using the Gram matrix of a hidden layer's activations. However, we get even better results combining this representation from multiple different layers. This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.
Minimizing the style cost will cause the image  G  to follow the style of the image  S .

3.3 - Defining the total cost to optimize
Finally, let's create a cost function that minimizes both the style and the content cost. The formula is:

J(G)=αJcontent(C,G)+βJstyle(S,G)

Exercise: Implement the total cost function which includes both the content cost and the style cost.

(see pyfile)

Expected Output:

J	35.34667875478276

What you should remember:

The total cost is a linear combination of the content costJcontent(C,G) and the style cost Jstyle(S,G)
α and β are hyperparameters that control the relative weighting between content and style

4 - Solving the optimization problem
Finally, let's put everything together to implement Neural Style Transfer!

Here's what the program will have to do:

Create an Interactive Session
Load the content image
Load the style image
Randomly initialize the image to be generated
Load the VGG16 model
Build the TensorFlow graph:
Run the content image through the VGG16 model and compute the content cost
Run the style image through the VGG16 model and compute the style cost
Compute the total cost
Define the optimizer and the learning rate
Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.
Lets go through the individual steps in detail.

You've previously implemented the overall cost J(G). We'll now set up TensorFlow to optimize this with respect to G. To do so, your program has to reset the graph and use an "Interactive Session". Unlike a regular session, the "Interactive Session" installs itself as the default session to build a graph. This allows you to run variables without constantly needing to refer to the session object, which simplifies the code.

Lets start the interactive session.

(see pyfile)

Let's load, reshape, and normalize our "content" image (the Louvre museum picture):
(see pyfile)

Let's load, reshape and normalize our "style" image (Claude Monet's painting):
(see pyfile)

Now, we initialize the "generated" image as a noisy image created from the content_image. By initializing the pixels of the generated image to be mostly noise but still slightly correlated with the content image, this will help the content of the "generated" image more rapidly match the content of the "content" image. (Feel free to look in nst_utils.py to see the details of generate_noise_image(...); to do so, click "File-->Open..." at the upper-left corner of this Jupyter notebook.)
(see pyfile)

Next, as explained in part (2), let's load the VGG16 model.
(see pyfile)

To get the program to compute the content cost, we will now assign a_C and a_G to be the appropriate hidden layer activations. We will use layer conv4_2 to compute the content cost. The code below does the following:

Assign the content image to be the input to the VGG model.
Set a_C to be the tensor giving the hidden layer activation for layer "conv4_2".
Set a_G to be the tensor giving the hidden layer activation for the same layer.
Compute the content cost using a_C and a_G.(see pyfile)
(see pyfile)

Note: At this point, a_G is a tensor and hasn't been evaluated. It will be evaluated and updated at each iteration when we run the Tensorflow graph in model_nn() below.
(see pyfile)

Exercise: Now that you have J_content and J_style, compute the total cost J by calling total_cost(). Use alpha = 10 and beta = 40.
(see pyfile)

You'd previously learned how to set up the Adam optimizer in TensorFlow. Lets do that here, using a learning rate of 2.0.
(see pyfile)

Exercise: Implement the model_nn() function which initializes the variables of the tensorflow graph, assigns the input image (initial generated image) as the input of the VGG16 model and runs the train_step for a large number of steps.
(see pyfile)

Run the following cell to generate an artistic image. It should take about 3min on CPU for every 20 iterations but you start observing attractive results after ≈140 iterations. Neural Style Transfer is generally trained using GPUs.
(see pyfile)

Expected Output:

Iteration 0 :	total cost = 5.05035e+09 
content cost = 7877.67 
style cost = 1.26257e+08
You're done! After running this, in the upper bar of the notebook click on "File" and then "Open". Go to the "/output" directory to see all the saved images. Open "generated_image" to see the generated image! :)

You should see something the image presented below on the right:

(refer images)
We didn't want you to wait too long to see an initial result, and so had set the hyperparameters accordingly. To get the best looking results, running the optimization algorithm longer (and perhaps with a smaller learning rate) might work better. After completing and submitting this assignment, we encourage you to come back and play more with this notebook, and see if you can generate even better looking images.

Here are few other examples:

The beautiful ruins of the ancient city of Persepolis (Iran) with the style of Van Gogh (The Starry Night)

(refer images)

The tomb of Cyrus the great in Pasargadae with the style of a Ceramic Kashi from Ispahan.

(refer images)

A scientific study of a turbulent fluid with the style of a abstract blue fluid painting.

(refer images)

Conclusion:

Great job on completing this assignment! You are now able to use Neural Style Transfer to generate artistic images. This is also your first time building a model in which the optimization algorithm updates the pixel values rather than the neural network's parameters. Deep learning has many different types of models and this is only one of them!

What you should remember:

Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image
It uses representations (hidden layer activations) based on a pretrained ConvNet.
The content cost function is computed using one hidden layer's activations.
The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
Optimizing the total cost function results in synthesizing new images.












