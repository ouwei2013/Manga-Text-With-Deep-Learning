
import argparse


parser = None
value = None

def is_defined(arg_name):
  if not value:
    return False
  if not getattr(value, arg_name):
    return False
  return True

def string_value(arg_name, default_value=""):
  if not value:
    return default_value
  try:
    argument = getattr(value, arg_name)
  except AttributeError:
    return default_value
  if not argument:
    return default_value
  elif type(argument)!=str:
    return default_value
  else:
    return argument

def boolean_value(arg_name, default_value=False):
  if not value:
    return default_value
  try:
    argument = getattr(value, arg_name)
  except AttributeError:
    return default_value
  if not argument:
    return default_value
  elif type(argument)!=bool:
    return default_value
  else:
    return argument

def integer_value(arg_name, default_value=0):
  if not value:
    return default_value
  try:
    argument = getattr(value, arg_name)
  except AttributeError:
    return default_value
  if not argument:
    return default_value
  elif type(argument)!=int:
    return default_value
  else:
    return argument

def float_value(arg_name, default_value=0.0):
  if not value:
    return default_value
  try:
    argument = getattr(value, arg_name)
  except AttributeError:
    return default_value
  if not argument:
    return default_value
  elif type(argument)!=float:
    return default_value
  else:
    return argument