#Content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
import os

import logging
import tensorflow as tf
import math

def calc_mem(linear):
    try:
        size_bytes = ((linear.in_features * linear.out_features) * (4 / (1024^3)))*2
    except:
        size_bytes = ((linear.in1_features * linear.in2_features* linear.out_features) * (4 / (1024^3)))*2
    return size_bytes

def format_mem(size_bytes):
    if size_bytes == 0:
           return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    print("%s %s" % (s, size_name[i]))
    return size_bytes

def print_results(article, abstract, decoded_output):
  print ("")
  print('ARTICLE:  %s', article)
  print('REFERENCE SUMMARY: %s', abstract)
  print('GENERATED SUMMARY: %s', decoded_output)
  print( "")


def make_html_safe(s):
  s.replace(b"<", b"&lt;")
  s.replace(b">", b"&gt;")
  return s


def rouge_eval(ref_dir, dec_dir):
  import pyrouge
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  print(log_str)
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  print("Writing final ROUGE results to %s..."%(results_file))
  with open(results_file, "w") as f:
    f.write(log_str)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  with summary_writer.as_default():
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    tf.summary.scalar(
      tag_name, running_avg_loss, step=step, description=None
    )
  return running_avg_loss


def write_for_rouge(reference_sents, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir):
  decoded_sents = []
  while len(decoded_words) > 0:
    try:
      fst_period_idx = decoded_words.index(".")
    except ValueError:
      fst_period_idx = len(decoded_words)
    sent = decoded_words[:fst_period_idx + 1]
    decoded_words = decoded_words[fst_period_idx + 1:]
    decoded_sents.append(b' '.join(sent))

  # pyrouge calls a perl script that puts the data into HTML files.
  # Therefore we need to make our output HTML safe.
  decoded_sents = [make_html_safe(w) for w in decoded_sents]
  reference_sents = [make_html_safe(w) for w in reference_sents]

  ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
  decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

  with open(ref_file, "w") as f:
    for idx, sent in enumerate(reference_sents):
      sent = sent.decode()
      f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
  with open(decoded_file, "w") as f:
    for idx, sent in enumerate(decoded_sents):
      sent = sent.decode()
      f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

  #print("Wrote example %i to file" % ex_index)
