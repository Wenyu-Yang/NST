import tensorflow as tf
import scipy.misc
from nst_utils import load_vgg_model, generate_noise_image, reshape_and_normalize_image, save_image


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G
    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[m, -1, n_C]), perm=[0, 1, 2])
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[m, -1, n_C]), perm=[0, 1, 2])
    
    # compute the cost
    J_content = (1./4./n_C/n_H/n_W) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)), axis=None)
    
    return J_content

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]), perm=[1, 0])
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]), perm=[1, 0])

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = (1./((2*n_C*n_H*n_W)**2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)), axis=None)
    
    return J_style_layer

def compute_style_cost(sess, model):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0
    
    STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]
    
    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha, beta):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha * J_content + beta * J_style
    
    return J

def graph(sess, content_image_path, style_image_path, model_path, alpha, beta):
    
    # Content image
    content_image = scipy.misc.imread(content_image_path)
    content_image = reshape_and_normalize_image(content_image)
    
    # Style image
    style_image = scipy.misc.imread(style_image_path)
    style_image = reshape_and_normalize_image(style_image)

    # Generated image correlated with content image
    generated_image = generate_noise_image(content_image)

    # Load pre_trained VGG model
    model = load_vgg_model(model_path)

    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model['conv4_2']
    
    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)
    
    # Set a_G to be the hidden layer activation from same layer. 
    a_G = out
    
    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)
    
    # Assign the input of the model to be the "style" image 
    sess.run(model['input'].assign(style_image))
    
    # Compute the style cost
    J_style = compute_style_cost(sess, model)

    # Total cost    
    J = total_cost(J_content, J_style, alpha, beta)
    
    # define optimizer
    optimizer = tf.train.AdamOptimizer(2.0)
    
    # define train_step
    train_step = optimizer.minimize(J)

    return model, generated_image, train_step, J, J_content, J_style

def model_nn(sess, content_image_path, style_image_path, model_path, 
             num_iterations = 200, alpha = 10, beta = 40):
    
    model, generated_image, train_step, J, J_content, J_style = graph(sess, content_image_path, style_image_path, model_path, alpha, beta)
    
    # Initialize global variables
    sess.run(tf.global_variables_initializer())
    
    # Run the initial generated image through the model.
    sess.run(model['input'].assign(generated_image))
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image
