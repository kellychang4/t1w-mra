import tensorflow as tf

def unet2D(
    batch_size,
    input_shape,
    n_base_filters = 16, 
    name = "unet2D"
):
    
    # Model Arguments
    conv_kwargs = {
        "kernel_size": (3, 3),
        "padding": "same", 
        "data_format": "channels_last"
    }
    
    conv_trans_kwargs = {
        "kernel_size": (2, 2), 
        "strides": 2,
        "padding": "same"
    }
    
    # Define Model Input Shapes
    inputs = tf.keras.Input(shape = input_shape, batch_size = batch_size)

    # Encoder_1: Layer 1
    x = tf.keras.layers.Conv2D(n_base_filters, **conv_kwargs)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    
    x = tf.keras.layers.Conv2D(n_base_filters * 2, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    skip_1 = x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    # Encoder_2: Layer 2
    x = tf.keras.layers.Conv2D(n_base_filters * 2, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(n_base_filters * 4, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    skip_2 = x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    # Encoder_3: Layer 3
    x = tf.keras.layers.Conv2D(n_base_filters * 4, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(n_base_filters * 8, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    skip_3 = x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    # Encoder_4: Layer 4
    x = tf.keras.layers.Conv2D(n_base_filters * 8, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(n_base_filters * 16, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Decoder_1: Layer 3
    x = tf.keras.layers.Conv2DTranspose(n_base_filters * 16, **conv_trans_kwargs)(x)
    x = tf.keras.layers.Concatenate(axis = -1)([skip_3, x])
    
    x = tf.keras.layers.Conv2D(n_base_filters * 8, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(n_base_filters * 8, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Decoder_2: Layer 2
    x = tf.keras.layers.Conv2DTranspose(n_base_filters * 8, **conv_trans_kwargs)(x)
    x = tf.keras.layers.Concatenate(axis = -1)([skip_2, x])
    
    x = tf.keras.layers.Conv2D(n_base_filters * 4, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(n_base_filters * 4, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Decoder_3: Layer 1
    x = tf.keras.layers.Conv2DTranspose(n_base_filters * 4, **conv_trans_kwargs)(x)
    x = tf.keras.layers.Concatenate(axis = -1)([skip_1, x])
    
    x = tf.keras.layers.Conv2D(n_base_filters * 2, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(n_base_filters * 2, **conv_kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Model Output Layer (reduce to same as input channels)
    outputs = tf.keras.layers.Conv2D(1, kernel_size = (1, 1))(x)
    
    # Define the Full Model
    model = tf.keras.Model(inputs = inputs, outputs = outputs, name = name)

    return model


def RUNet(
  batch_size, 
  input_shape, 
  name = "RUNet"
): 

  conv_kwargs = {
    "padding": "same", 
    "data_format": "channels_last"
  }

  """Define Residual Block"""
  def residual_block(x_in, filters, kernel_size, **kwargs):
    x = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, **kwargs)(x_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, **kwargs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if x_in.shape != x.shape: # if input shape is not output shape, projection!
        x_in = tf.keras.layers.Conv2D(filters, kernel_size = 1)(x_in)
    return tf.keras.layers.Add()([x, x_in])

  """Define Upsampling Block"""
  def upsample_block(x, filters, kernel_size, **kwargs):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, **kwargs)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, **kwargs)(x)
    x = tf.keras.layers.Activation("relu")(x)
    return tf.keras.layers.Activation("relu")(x)


  # Define Model Input Layer
  inputs = tf.keras.Input(shape = input_shape, batch_size = batch_size)

  # Block 1: Encoder - Level 1
  x = tf.keras.layers.Conv2D(64, kernel_size = 7, **conv_kwargs)(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = skip_1 = tf.keras.layers.Activation("relu")(x)

  # Block 2: Encoder - Level 2
  x = tf.keras.layers.MaxPooling2D()(x) # might need to change "pool_size"
  x = residual_block(x, filters = 64, kernel_size = 3, **conv_kwargs)
  x = residual_block(x, filters = 64, kernel_size = 3, **conv_kwargs)
  x = residual_block(x, filters = 64, kernel_size = 3, **conv_kwargs)
  x = skip_2 = residual_block(x, filters = 128, kernel_size = 3, **conv_kwargs)

  # Block 3: Encoder - Level 3
  x = tf.keras.layers.MaxPooling2D()(x) # might need to change "pool_size"
  x = residual_block(x, filters = 128, kernel_size = 3, **conv_kwargs)
  x = residual_block(x, filters = 128, kernel_size = 3, **conv_kwargs)
  x = residual_block(x, filters = 128, kernel_size = 3, **conv_kwargs)
  x = skip_3 = residual_block(x, filters = 256, kernel_size = 3, **conv_kwargs)

  # Block 4: Encoder - Level 4
  x = tf.keras.layers.MaxPooling2D()(x) # might need to change "pool_size"
  x = residual_block(x, filters = 256, kernel_size = 3, **conv_kwargs)
  x = residual_block(x, filters = 256, kernel_size = 3, **conv_kwargs)
  x = residual_block(x, filters = 256, kernel_size = 3, **conv_kwargs)
  x = residual_block(x, filters = 256, kernel_size = 3, **conv_kwargs)
  x = residual_block(x, filters = 256, kernel_size = 3, **conv_kwargs)
  x = skip_4 = residual_block(x, filters = 512, kernel_size = 3, **conv_kwargs)

  # Block 5: Encoder - Level 5
  x = tf.keras.layers.MaxPooling2D()(x) # might need to change "pool_size"
  x = residual_block(x, filters = 512, kernel_size = 3, **conv_kwargs)
  x = residual_block(x, filters = 512, kernel_size = 3, **conv_kwargs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = skip_5 = tf.keras.layers.Activation("relu")(x)

  # Block 6: Bridge
  x = tf.keras.layers.Conv2D(1024, kernel_size = 3, **conv_kwargs)(x)
  x = tf.keras.layers.Activation("relu")(x)
  x = tf.keras.layers.Conv2D(512, kernel_size = 3, **conv_kwargs)(x)
  x = tf.keras.layers.Activation("relu")(x)

  # Block 7: Decoder - Level 5
  x = tf.keras.layers.Concatenate()([skip_5, x])
  x = upsample_block(x, filters = 512, kernel_size = 3, **conv_kwargs)

  # Block 8: Decoder - Level 4
  x = tf.nn.depth_to_space(x, block_size = 2)
  x = tf.keras.layers.Concatenate()([skip_4, x])
  x = upsample_block(x, filters = 384, kernel_size = 3, **conv_kwargs)

  # Block 9: Decoder - Level 3
  x = tf.nn.depth_to_space(x, block_size = 2)
  x = tf.keras.layers.Concatenate()([skip_3, x])
  x = upsample_block(x, filters = 256, kernel_size = 3, **conv_kwargs)

  # Block 10: Decoder - Level 2
  x = tf.nn.depth_to_space(x, block_size = 2)
  x = tf.keras.layers.Concatenate()([skip_2, x])
  x = upsample_block(x, filters = 96, kernel_size = 3, **conv_kwargs)

  # Block 11: Decoder - Level 1
  x = tf.nn.depth_to_space(x, block_size = 2)
  x = tf.keras.layers.Concatenate()([skip_1, x])
  x = tf.keras.layers.Conv2D(99, kernel_size = 3, **conv_kwargs)(x)
  x = tf.keras.layers.Activation("relu")(x)
  x = tf.keras.layers.Conv2D(99, kernel_size = 3, **conv_kwargs)(x)
  x = tf.keras.layers.Activation("relu")(x)

  # Block 12: Output
  x = tf.keras.layers.Conv2D(3, kernel_size = 1, **conv_kwargs)(x)
  outputs = tf.keras.layers.Activation("relu")(x)

  # Define the Full Model
  model = tf.keras.Model(inputs = inputs, outputs = outputs, name = name)

  return model