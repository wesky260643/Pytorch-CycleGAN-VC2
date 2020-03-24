#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File Name : logger.py
# @Purpose :
# @Creation Date : 2020-03-19 11:48:57
# @Last Modified : 2020-03-19 11:51:46
# @Created By :  chenjiang
# @Modified By :  chenjiang

from tensorboardX import SummaryWriter


import functools
print = functools.partial(print, flush=True)


class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)



