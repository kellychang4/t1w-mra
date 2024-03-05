import tensorflow as tf


class PerceptualLossVGG16(tf.keras.losses.Loss):
    def __init__(self, loss_layer, input_shape):
        super().__init__(
            reduction = tf.keras.losses.Reduction.AUTO, 
            name      = "Perceptual_Loss_with_VGG16"
        )
        
        self.loss_layer  = loss_layer
        self.input_shape = input_shape
        
        vgg16 = tf.keras.applications.vgg16.VGG16(
            include_top = False,
            weights     = "imagenet",
            input_shape = input_shape,
            pooling     = "max"
        )
        vgg16.trainable = False
        
        self.loss_model = tf.keras.Model(
            inputs  = vgg16.input, 
            outputs = vgg16.get_layer(loss_layer).output
        )
        
    @tf.function
    def call(self, y_true, y_pred):
        # Compute Model Loss for Predicted and Actual Images
        loss_pred = tf.cast(self.loss_model(y_pred), tf.float32)
        loss_true = tf.cast(self.loss_model(y_true), tf.float32)

        # Compute Nomalized Sums of Squared Error
        n    = tf.cast(tf.size(loss_pred), dtype = tf.float32)
        loss = tf.square(tf.norm(loss_pred - loss_true, ord = 2))
        return tf.divide(loss, n)
    
    def get_config(self):
        config = {
            "loss_layer":  self.loss_layer,
            "input_shape": self.input_shape,
            "loss_model":  self.loss_model,
        }
        super_config = super().get_config()
        return {**super_config, **config}
    

class PerceptualLossVGG19(tf.keras.losses.Loss):
    def __init__(self, loss_layer, input_shape):
        super().__init__(
            reduction = tf.keras.losses.Reduction.AUTO, 
            name      = "Perceptual_Loss_with_VGG19"
        )
        
        self.loss_layer  = loss_layer
        self.input_shape = input_shape
        
        vgg19 = tf.keras.applications.vgg19.VGG19(
            include_top = False,
            weights     = "imagenet",
            input_shape = input_shape,
            pooling     = "max"
        )
        vgg19.trainable = False
        
        self.loss_model = tf.keras.Model(
            inputs  = vgg19.input, 
            outputs = vgg19.get_layer(loss_layer).output
        )
        
    @tf.function
    def call(self, y_true, y_pred):
        # Compute Model Loss for Predicted and Actual Images
        loss_pred = tf.cast(self.loss_model(y_pred), tf.float32)
        loss_true = tf.cast(self.loss_model(y_true), tf.float32)

        # Compute Nomalized Sums of Squared Error
        n    = tf.cast(tf.size(loss_pred), dtype = tf.float32)
        loss = tf.square(tf.norm(loss_pred - loss_true, ord = 2))
        return tf.divide(loss, n)
    
    def get_config(self):
        config = {
            "loss_layer":  self.loss_layer,
            "input_shape": self.input_shape,
            "loss_model":  self.loss_model,
        }
        super_config = super().get_config()
        return {**super_config, **config}


@tf.function
def SSIMLoss(y_true, y_pred):
    """Structural Similarity Loss Function. Requires [0.0, 1.0] inputs."""
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
