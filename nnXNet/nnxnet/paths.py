#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

nnXNet_raw = os.environ.get('nnXNet_raw')
nnXNet_preprocessed = os.environ.get('nnXNet_preprocessed')
nnXNet_results = os.environ.get('nnXNet_results')

if nnXNet_raw is None:
    print("nnXNet_raw is not defined and nnX-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnX-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if nnXNet_preprocessed is None:
    print("nnXNet_preprocessed is not defined and nnX-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if nnXNet_results is None:
    print("nnXNet_results is not defined and nnX-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
