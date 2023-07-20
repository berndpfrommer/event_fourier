# -----------------------------------------------------------------------------
# Copyright 2022 Bernd Pfrommer <bernd.pfrommer@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
import audio_common_msgs
import time
from bag_reader import BagReader
import argparse
import numpy as np
import os


def main(args):
    bag = BagReader(args.bagfile, args.topic)
    t0 = time.time()
    num_msgs = 0
    os.remove(args.output_file)
    outfile = open(args.output_file, 'a')
    print(f"writing to file {args.output_file}")

    while bag.has_next():
        topic, msg, t_rec = bag.read_next()
        if isinstance(msg, audio_common_msgs.msg.AudioData):
            arr = np.array(msg.data)
            arr.tofile(outfile)
            num_msgs = num_msgs + 1
    t1 = time.time()
    dt = t1 - t0
    print(f'took {dt:.3f}s to read {num_msgs} audio',
          f' packets ({num_msgs / dt:.3f} packets/s)')
    outfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert bag file to wav.')
    parser.add_argument('--output_file', '-o', action='store',
                        default='audio.wav', help='output wav file')
    parser.add_argument('--topic', '-t', action='store',
                        default='/audio/audio', help='topic name')
    parser.add_argument('bagfile')

    args = parser.parse_args()
    main(args)
    
