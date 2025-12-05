#!/bin/bash
xvfb-run -a -s "-screen 0 1024x768x24" /usr/bin/java "$@"
