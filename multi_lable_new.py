import os
import time
from glob import glob
import pandas as pd
import tensorflow as tf
import numpy as np

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=331, input_width=331,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
  model_file = "tf_files/graph_files/output_graph.pb"
  label_file = "tf_files/graph_files/output_labels.txt"
  input_height = 331
  input_width = 331
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"

  graph = load_graph(model_file)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  labels = load_labels(label_file)

  # Load label file
  labels = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
      labels.append(l.rstrip())

  mypath = "test_set/Testing_Set_Traffic"
  true = glob((mypath + "/*"))
  print('true:', true)
  newDF = pd.DataFrame(columns=['Original Label', 'Filename', 'Primary Result', 'Primary Score',
                                'Secondary Result', 'Secondary Score', 'Tertiary Result', 'Tertiary Score'])
  for name in true:
    lb= os.path.basename(name)
    print('Looking Images in',lb)
    files = os.listdir(name)
    i = 0
    for filename in files:
        file_name = os.path.join(name,filename)
        print('file name:', str(file_name))
        file_name = str(file_name)
        i = i+1
        if i%10 == 0 :
            print("Processed",i,"images")
        try:
            t = read_tensor_from_image_file(file_name,
                                            input_height=input_height,
                                            input_width=input_width,
                                            input_mean=input_mean,
                                            input_std=input_std)
            with tf.Session(graph=graph) as sess:
                start = time.time()
                results = sess.run(output_operation.outputs[0],
                                   {input_operation.outputs[0]: t})
                end = time.time()
            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]

            primary_score = results[top_k[0]]
            secondary_score = results[top_k[1]]
            tertiary_score = results[top_k[2]]

            primary_label = labels[top_k[0]]
            secondary_label = labels[top_k[1]]
            tertiary_label = labels[top_k[2]]

            newDF = newDF.append({'Original Label': lb, 'Filename': os.path.basename(file_name),
                                  'Primary Result': primary_label, 'Primary Score': primary_score,
                                  'Secondary Result': secondary_label, 'Secondary Score': secondary_score,
                                  'Tertiary Result': tertiary_label, 'Tertiary Score': tertiary_score},
                                 ignore_index=True)
        except:
            print('There was some problem in Image data file',os.path.basename(file_name))
            continue

  output_file = 'ClassificationResult.xlsx'
  print("Output file name given as:", output_file)
  writer = pd.ExcelWriter(output_file)
  newDF.to_excel(writer, 'Sheet1')
  writer.save()