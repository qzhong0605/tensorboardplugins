/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
namespace tf.convert.controls {

interface DeviceNameExclude {
  regex: RegExp,
}

const DEVICE_NAME_REGEX = /device:([^:]+:[0-9]+)$/;

/**
 * Display only devices matching one of the following regex.
 */
const DEVICE_NAMES_INCLUDE: DeviceNameExclude[] = [
  {
    // Don't include GPU stream, memcpy, etc. devices
    regex: DEVICE_NAME_REGEX,
  }
];

interface StatsDefaultOff {
  regex: RegExp,
  msg: string, // 'Excluded by default since...'
}

/**
 * Stats from device names that match these regexes will be disabled by default.
 * The user can still turn on a device by selecting the checkbox in the device list.
 */
const DEVICE_STATS_DEFAULT_OFF: StatsDefaultOff[] = [];

export interface Selection {
  run: string,
  tag: string | null,
  type: tf.convert.SelectionType,
}

export interface DeviceForStats {
  [key: string]: boolean;
}

// TODO(stephanwlee): Move this to tf-convert-dashboard
export interface TagItem {
  tag: string | null,
  displayName: string,
  conceptualGraph: boolean,
  opGraph: boolean,
  profile: boolean,
}

// TODO(stephanwlee): Move this to tf-convert-dashboard
export interface RunItem {
  name: string,
  tags: TagItem[],
}

// TODO(stephanwlee): Move this to tf-convert-dashboard
export type Dataset = Array<RunItem>;

interface CurrentDevice {
  device: string,
  suffix: string,
  used: boolean,
  ignoredMsg: string | null,
}

export enum ColorBy {
  COMPUTE_TIME = 'compute_time',
  MEMORY = 'memory',
  STRUCTURE = 'structure',
  XLA_CLUSTER = 'xla_cluster',
  OP_COMPATIBILITY = 'op_compatibility',
}

interface ColorParams {
  minValue: number,
  maxValue: number,
  // HEX value describing color.
  startColor: string,
  // HEX value describing color.
  endColor: string,
}

interface DeviceColor {
  device: string;
  color: string;
}

interface XlaClusterColor {
  xla_cluster: string;
  color: string;
}

// TODO(stephanwlee) Move this to tf-convert.html when it becomes TypeScript.
interface ColorByParams {
  compute_time: ColorParams,
  memory: ColorParams,
  device: DeviceColor[],
  xla_cluster: XlaClusterColor[],
}

const GRADIENT_COMPATIBLE_COLOR_BY: Set<ColorBy> = new Set([
    ColorBy.COMPUTE_TIME, ColorBy.MEMORY]);


Polymer({
  is: 'tf-convert-controls',
  properties: {
    // TODO:
    statistics:{
      type: Array,
      value:[],
    },
    selectedGraph:String,
    graph: Object,
    graphHierarchy: {
      type: Object,
      notify: true,
    },
    param:{
      type: Object,
    },
    srcPath: String,
    desPath: String,
    srcType: {
      type: Number,
      observer:'_srcTypeChanged'
    },
    desType: {
      type: Number,
      observer:'_desTypeChanged'
    },
    _modelTypes:{
      type: Array,
      value: ['onnx', 'caffe', 'caffe2', 'torch', 'tf'],
    },
    // Public API.
    /**
     * @type {?tf.convert.proto.StepStats}
     */
    stats: {
      value: null,
      type: Object,
      // observer: '_statsChanged',
    },
    
    /**
     * @type {!tf.convert.controls.ColorBy}
     */
    colorBy: {
      type: String,
      value: ColorBy.STRUCTURE,
      notify: true,
    },
    colorByParams: {
      type: Object,
      notify: true,
      // TODO(stephanwlee): Change readonly -> readOnly and fix the setter.
      readonly: true,
    },
    /**
     * @type {!Selection}
     */
    selection: {
      type: Object,
      notify: true,
    },
    targetParams: {
      type: Object,
      notify: true,
    },
    left_width:{
      type: Number,
      value: 40,
    },
    right_width:{
      type: Number,
      value: 40,
    }
  },

  listeners: {},

  ready: function(){},
  _srcTypeChanged: function(){
    if(this.srcType == 2){
      document.getElementById('srcnotc2').style.display = 'none'
      document.getElementById('srcisc2').style.display = ''
    }else{
      document.getElementById('srcnotc2').style.display = ''
      document.getElementById('srcisc2').style.display = 'none'
    }
  },
  _desTypeChanged: function(){
    if(this.desType == 2){
      document.getElementById('desnotc2').style.display = 'none'
      document.getElementById('desisc2').style.display = ''
    }else{
      document.getElementById('desnotc2').style.display = ''
      document.getElementById('desisc2').style.display = 'none'
    }
  },
  dragLeft(){
    var left = document.getElementsByClassName('source-graphboard') as HTMLCollectionOf<HTMLElement>;
    var right = document.getElementsByClassName('target-graphboard') as HTMLCollectionOf<HTMLElement>;
    // left[0].style.width = '20%'
    this.left_width -= 2
    this.right_width += 2 
    left[0].style.width = this.left_width.toString() + '%'
    right[0].style.width = this.right_width.toString() + '%'
    // console.info(left[0],right[0])
    // console.info('left')
  },
  dragRight(){
    var left = document.getElementsByClassName('source-graphboard') as HTMLCollectionOf<HTMLElement>;
    var right = document.getElementsByClassName('target-graphboard') as HTMLCollectionOf<HTMLElement>;
    // left[0].style.width = '20%'
    this.left_width += 2
    this.right_width -= 2 
    left[0].style.width = this.left_width.toString() + '%'
    right[0].style.width = this.right_width.toString() + '%'
    // console.info('right')
  },

  transformModel: function(){
    var destination_type = this._modelTypes[this.desType]
    var data = {}
    if(destination_type == 'caffe2'){
      data = {
        'predict_net': (<HTMLInputElement>document.getElementById('predict_net_des')).value,
        'init_net': (<HTMLInputElement>document.getElementById('init_net_des')).value,
        'destination_type': destination_type,
      }
    }else{
      data = {
        'destination_path': this.desType,
        'destination_type': destination_type,
      }
    }

    this.targetParams = data
  },

  loadModel: function(){
    var source_type = this._modelTypes[this.srcType]
    var data = {}
    if(source_type == 'caffe2'){
      data = {
        'predict_net': (<HTMLInputElement>document.getElementById('predict_net')).value,
        'init_net': (<HTMLInputElement>document.getElementById('init_net')).value,
        'source_type': source_type,
      }
    }
    else{
      data = {
        'source_path': this.srcPath,
        'source_type': source_type,
      }
    }
    
    this.selection = data
  },

  streamParse: function(
    arrayBuffer: ArrayBuffer, callback: (string) => void,
    chunkSize: number = 1000000, delim: string = '\n'): Promise<boolean> {
    return new Promise<boolean>(function(resolve, reject) {
      function readChunk(oldData: string, newData: string, offset: number) {
        const doneReading = offset >= arrayBuffer.byteLength;
        const parts = newData.split(delim);
        parts[0] = oldData + parts[0];

        // The last part may be part of a longer string that got cut off
        // due to the chunking.
        const remainder = doneReading ? '' : parts.pop();

        for (let part of parts) {
          try {
            callback(part);
          } catch (e) {
            reject(e);
            return;
          }
        }

        if (doneReading) {
          resolve(true);
          return;
        }

        const nextChunk = new Blob([arrayBuffer.slice(offset, offset + chunkSize)]);
        const file = new FileReader();
        file.onload = function(e: any) {
          readChunk(remainder, e.target.result, offset + chunkSize);
        };
        file.readAsText(nextChunk);
      }

      readChunk('', '', 0);
    });
  },

  parseValue(value: string): string|number|boolean {
    if (value === 'true') {
      return true;
    }
    if (value === 'false') {
      return false;
    }
    let firstChar = value[0];
    if (firstChar === '"') {
      return value.substring(1, value.length - 1);
    }
    let num = parseFloat(value);
    return isNaN(num) ? value : num;
  },

  getStatistics: function(){
    var mthis = this
    var path = tf_backend.getRouter().pluginRoute('convert', '/statistics', );
    fetch(path).then((res) => {
      // Fetch does not reject for 400+.
      if (res.ok) {
        res.arrayBuffer().then(function(arrayBuffer: ArrayBuffer){
          if(arrayBuffer!=null){
            var output = [];
            var tmp = {};
            mthis.streamParse(arrayBuffer, function(line: String){
              if(line){
                line = line.trim();
                switch(line[line.length-1]){
                  case '{':
                    tmp = {};
                    break;
                  case '}':
                    output.push(tmp);
                    break;
                  default:
                    var index = line.indexOf(':');
                    var k = line.substring(0, index);
                    var value = mthis.parseValue(line.substring(index + 2).trim());
                    tmp[k] = value;
                    break;
                }
              }
            }).then(function(){
              mthis.statistics = output
            })
          }
        })
      }
    });
  },
 
  

});

}  // namespace tf.convert.controls
