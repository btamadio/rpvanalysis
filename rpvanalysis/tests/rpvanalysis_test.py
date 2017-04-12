#!/usr/bin/env python
from nose.tools import *
import rpvanalysis

def setup():
    print "SETUP!"
    
def teardown():
    print "TEAR DOWN!"

def test_basic():
    a=rpvanalysis.analyzer()
    print "I RAN!"
