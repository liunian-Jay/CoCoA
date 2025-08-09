import re
import string
from collections import Counter

""" For direct call """
def get_evaluation(task, res_answer, gold_answers, regex=True, gold_answer_list=None):
  if task == 'ARC_Challenge':
      score = accuracy(res_answer, gold_answers)
      return score, 0 
  elif task == 'ASQA':
      print("x"*100)
      score = ASQA_EM(res_answer, gold_answer_list, regex=regex)
      f1_score = f1_max_over_ground_truths(res_answer, gold_answers)
      return score, f1_score
  else:
      score = em_max_over_ground_truths(res_answer, gold_answers, regex=regex)
      f1_score = f1_max_over_ground_truths(res_answer, gold_answers)
      return score, f1_score
  



###############################################
###############################################
###############################################
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
  """ EM: ground_truth is a str """
  return normalize_answer(prediction) == normalize_answer(ground_truth)

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            normalize_answer(pattern),
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(normalize_answer(text)) is not None
    
def f1_score(prediction, ground_truth):
  """ F1: ground_truth is a str """
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
      return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1



def em_max_over_ground_truths(prediction, ground_truths, regex=True):
  """ EM: ground_truth is a list """
  return max([regex_match(prediction, gt) if regex else exact_match_score(prediction, gt) for gt in ground_truths])

def f1_max_over_ground_truths(prediction, ground_truths):
  """ F1: ground_truth is a list """
  return max([f1_score(prediction, gt) for gt in ground_truths])


def accuracy(prediction, ground_truths):
  """ ACC  """
  if type(prediction) is str and (prediction[0] == "#" or prediction[0] == ":") and len(prediction)>=1:
    prediction = prediction[1:]
  if len(prediction)>1:
    prediction = prediction.split(':')[0]
  if len(prediction)>1:
    prediction = prediction.split('.')[0]
  
  # print(prediction)

  for gt in ground_truths:
    if gt == prediction:
        return 1
  return 0

# text ='Jamaican people speak Jamaican Patois, English, and Jamaican Sign Language.'
# pattern = 'Jamaican English'
# print(regex_match(text, pattern))
# print(regex_match(pattern, text))


def ASQA_EM(prediction, ground_truths_list, regex=True):
  for ground_truths in ground_truths_list:
    if not max([regex_match(prediction, gt) if regex else exact_match_score(prediction, gt) for gt in ground_truths]):
       return False
  return  True