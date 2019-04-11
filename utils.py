import os

root_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(root_dir, 'models')
object_cache_dir = os.path.join(root_dir, '.ml-sandbox-object-cache')

def list_models():
  models = os.listdir(models_dir)
  models = filter(lambda x: x.endswith('.py'), models)
  models = map(lambda x: x.replace('.py', ''), models)

  return list(models)

def save_model(model, params):
  if not os.path.exists(object_cache_dir):
    os.mkdir(object_cache_dir)

  base_name = '{name}__val_{val:.5f}__lr_{lr}__decay_{decay}'.format(
    name=model.name,
    val=model.history.history['val_categorical_accuracy'][-1],
    lr=params.learning_rate,
    decay=params.learning_rate_decay
  )

  file_name = base_name.replace('.', '') + '.hdf5'

  file_path = os.path.join(object_cache_dir, file_name)

  model.save_weights(file_path)

  print('Saved weights to "{filepath}"'.format(filepath=file_path))

  model_config_filename = os.path.join(object_cache_dir, model.name + '.yml')

  if os.path.exists(model_config_filename):
    return
  
  with open(model_config_filename, 'w') as fd:
    fd.write(model.to_yaml())
    fd.close()

  print('Saved model config to "{filepath}"'.format(filepath=model_config_filename))
