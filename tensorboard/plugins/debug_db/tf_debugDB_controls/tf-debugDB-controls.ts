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
namespace tf.graph.controls {

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
    type: tf.graph.SelectionType,
  }
  
  export interface DeviceForStats {
    [key: string]: boolean;
  }
  
  // TODO(stephanwlee): Move this to tf-debugDB-dashboard
  export interface TagItem {
    tag: string | null,
    displayName: string,
    conceptualGraph: boolean,
    opGraph: boolean,
    profile: boolean,
  }
  
  // TODO(stephanwlee): Move this to tf-debugDB-dashboard
  export interface RunItem {
    name: string,
    tags: TagItem[],
  }
  
  // TODO(stephanwlee): Move this to tf-debugDB-dashboard
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
  
  // TODO(stephanwlee) Move this to tf-debugDB.html when it becomes TypeScript.
  interface ColorByParams {
    compute_time: ColorParams,
    memory: ColorParams,
    device: DeviceColor[],
    xla_cluster: XlaClusterColor[],
  }
  
  const GRADIENT_COMPATIBLE_COLOR_BY: Set<ColorBy> = new Set([
      ColorBy.COMPUTE_TIME, ColorBy.MEMORY]);
  
  
  Polymer({
    is: 'tf-debugDB-controls',
    properties: {
      machineidList:{
        type: Array,
        value: [],
      },
      selectedIdentification:{
        type: Number,
        observer:'_selectedIdentificationChanged',
      },
      attachList:{
        type: Array,
        value: [],
      },
      attachMap:{
        type: Object,
        value:{},
      },
      machineList:{
        type: Array,
        value: [],
        notify: true,
      },
      _srcMode:{
        type: Number,
        value: 2,
        observer:'_modeChanged'
      },
      _srcTypes:{
        type: Array,
        value: ['caffe2', 'caffe', 'onnx', 'torch', 'tf'],
      },
      _deviceTypes:{
        type: Array,
        value:['X86','ARM','CUDA','HIP']
      },
      selctedDeviceType:{
        type: Number,
      },
      newIsStop:{
        type: Boolean,
        value: false,
      },
      loadType: {
        type: Boolean,
        value: true,
      },
      loadTypeText:{
        type: String,
        computed:'_loadTypeChanged(loadType)'
      },
      deviceIds:{
        type: Array,
        value: [],
      },
      deviceId: String,
      // Public API.
      /**
       * @type {?tf.graph.proto.StepStats}
       */
      stats: {
        value: null,
        type: Object,
        // observer: '_statsChanged',
      },
      
      /**
       * @type {!tf.graph.controls.ColorBy}
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
      datasets: {
        type: Array,
        observer: '_datasetsChanged',
        value: () => [],
      },
      /**
       * @type {!Selection}
       */
      selection: {
        type: Object,
        notify: true,
        readOnly: true,
        computed: '_computeSelection(datasets, _selectedRunIndex, _selectedTagIndex, _selectedGraphType)',
      },
      
      _selectedRunIndex: {
        type: Number,
        value: 0,
        observer: '_selectedRunIndexChanged',
      },
      _selectedTagIndex: {
        type: Number,
        value: 0,
        observer: '_selectedTagIndexChanged',
      },
      /**
       * @type {tf.graph.SelectionType}
       */
      _selectedGraphType: {
        type: String,
        value: tf.graph.SelectionType.OP_GRAPH,
      },
    },
  
    listeners: {},
  
    ready: function(){},
    _modeChanged: function(){
      if(document.getElementById('not-caffe2')){
        if(this._srcMode){
          document.getElementById('not-caffe2').style.display = ''
          document.getElementById('is-caffe2').style.display = 'none'
        }else{
          document.getElementById('not-caffe2').style.display = 'none'
          document.getElementById('is-caffe2').style.display = ''
        }
      }
    },
    _loadTypeChanged: function(loadType){
      if(loadType){
        if(document.getElementById('newPage')){
          document.getElementById('newPage').style.display = ''
          document.getElementById('attachPage').style.display = 'none'
        }
        return 'New'
      }else{
        document.getElementById('newPage').style.display = 'none'
        document.getElementById('attachPage').style.display = ''
        return 'Attach'
      }
    },
    addMachine: function(){
      this.machineList.push({'m':'','id':''})
      var item = document.createElement("div")
      item.setAttribute('id','machine'+this.machineList.length)
      item.innerHTML = "<paper-input class='machine' id='machine' label='Machine' value=''></paper-input><paper-input class='id' id='id' label='Device Id' value=''></paper-input>"
      document.getElementById('machineList').appendChild(item)
    },
    newStart: function(){
      const params = new URLSearchParams();
      
      var model_type = this._srcTypes[this._srcMode];
      var file_type = (<HTMLInputElement>document.getElementById('fileType')).value;
      params.set('model_type', model_type);
      params.set('file_type', file_type);

      if(this._srcMode){
        var source_path = (<HTMLInputElement>document.getElementById('srcPath')).value;
        params.set('source_path', source_path);
      }else{
        var predict_net = (<HTMLInputElement>document.getElementById('predictNet')).value;
        var init_net = (<HTMLInputElement>document.getElementById('initNet')).value;
        params.set('predict_net', predict_net);
        params.set('init_net', init_net);
      }
      var batchSize = (<HTMLInputElement>document.getElementById('batchSize')).value;
      var memorySize = (<HTMLInputElement>document.getElementById('memorySize')).value;
      var optimizationMethod = (<HTMLInputElement>document.getElementById('optimizationMethod')).value;
      var learningRate = (<HTMLInputElement>document.getElementById('learningRate')).value;
      var totalIteration = (<HTMLInputElement>document.getElementById('totalIteration')).value;
      params.set('batch_size', batchSize);
      params.set('memory_size', memorySize);
      params.set('optimization_method', optimizationMethod);
      params.set('learning_rate', learningRate);
      params.set('total_iteration', totalIteration);
      
      var device_type = this._deviceTypes[this.selctedDeviceType];
      params.set('device_type', device_type);

      for(var i=1; i<=this.machineList.length; i++){
        var id = 'machine'+ i;
        var item = document.getElementById(id);
        this.machineList[i-1].m = (<HTMLInputElement>item.getElementsByClassName('machine')[0]).value;
        this.machineList[i-1].id = (<HTMLInputElement>item.getElementsByClassName('id')[0]).value;
      }
      params.set('machine_list', JSON.stringify(this.machineList));
      
      new Promise(() => {
        fetch(tf_backend.getRouter().pluginRoute('graphdebug', '/newstart', params)).then((res) => {
          // Fetch does not reject for 400+.
          if(res.ok){
            res.text().then(function(msg){
              (<HTMLInputElement>document.getElementById('fileType')).value = '';
              (<HTMLInputElement>document.getElementById('srcPath')).value = '';
              (<HTMLInputElement>document.getElementById('predictNet')).value = '';
              (<HTMLInputElement>document.getElementById('initNet')).value = '';
              (<HTMLInputElement>document.getElementById('batchSize')).value = '';
              (<HTMLInputElement>document.getElementById('memorySize')).value = '';
              (<HTMLInputElement>document.getElementById('optimizationMethod')).value = '';
              (<HTMLInputElement>document.getElementById('learningRate')).value = '';
              (<HTMLInputElement>document.getElementById('totalIteration')).value = '';
              document.getElementById('machineList').innerHTML = ""
            })
          }
        });
      });

    },
    newStop: function(){
      this.newIsStop = true
      const params = new URLSearchParams();
      new Promise(() => {
        fetch(tf_backend.getRouter().pluginRoute('graphdebug', '/newstop', params)).then((res) => {
          // Fetch does not reject for 400+.
          if(res.ok){
            res.text().then(function(msg){
              
            })
          }
        });
      });
    },
    newContinue: function(){
      this.newIsStop = false
      document.getElementById('cancel-bt').click()
      const params = new URLSearchParams();
      var iteration_number = (<HTMLInputElement>document.getElementById('iteration_number')).value;
      params.set('iteration_number', iteration_number);
      new Promise(() => {
        fetch(tf_backend.getRouter().pluginRoute('graphdebug', '/newcontinue', params)).then((res) => {
          // Fetch does not reject for 400+.
          if(res.ok){
            res.text().then(function(msg){
              
            })
          }
        });
      });
    },
    attach: function(){
      var mthis= this
      var attachList = []
      this.attachList.forEach(element => {
        attachList.push(element);
      });
      var network_identification = (<HTMLInputElement>document.getElementById('network_identification')).value;
      const params = new URLSearchParams();
      params.set('network_identification',network_identification);
      new Promise(() => {
        fetch(tf_backend.getRouter().pluginRoute('graphdebug', '/attach', params)).then((res) => {
          // Fetch does not reject for 400+.
          if(res.ok){
            res.json().then(function(respond){
              attachList.push(network_identification);
              mthis.attachList = attachList;
              mthis.attachMap[network_identification] = respond;
              (<HTMLInputElement>document.getElementById('network_identification')).value = ''
            })
          }
        });
      });
    },
    _selectedIdentificationChanged: function(){
      document.getElementById('attachInfo').style.display = '';
      var selectedIdentification = this.attachList[this.selectedIdentification];
      var attachInfo = this.attachMap[selectedIdentification];
      (<HTMLInputElement>document.getElementById('model_type')).value = attachInfo.model_type;
      this.machineidList = attachInfo.list
    },
    attachStop: function(){
      const params = new URLSearchParams();
      params.set('identification',this.attachList[this.selectedIdentification]);
      new Promise(() => {
        fetch(tf_backend.getRouter().pluginRoute('graphdebug', '/attachstop', params)).then((res) => {
          // Fetch does not reject for 400+.
          if(res.ok){
            res.text().then(function(msg){
              
            })
          }
        });
      });
    },
    attachContinue: function(){
      document.getElementById('cancel-bt').click()
      const params = new URLSearchParams();
      var iteration_number = (<HTMLInputElement>document.getElementById('iteration_number_2')).value;
      params.set('iteration_number', iteration_number);
      params.set('identification',this.attachList[this.selectedIdentification]);
      new Promise(() => {
        fetch(tf_backend.getRouter().pluginRoute('graphdebug', '/attachcontinue', params)).then((res) => {
          // Fetch does not reject for 400+.
          if(res.ok){
            res.text().then(function(msg){
              
            })
          }
        });
      });
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
    
  
    _datasetsChanged: function(newDatasets: Dataset, oldDatasets: Dataset) {
      if (oldDatasets != null) {
        // Select the first dataset by default.
        this._selectedRunIndex = 0;
      }
    },
  
    _computeSelection: function(datasets: Dataset, _selectedRunIndex: number,
        _selectedTagIndex: number, _selectedGraphType: tf.graph.SelectionType) {
      if (!datasets[_selectedRunIndex] ||
          !datasets[_selectedRunIndex].tags[_selectedTagIndex]) {
        return null;
      }
  
      return {
        run: datasets[_selectedRunIndex].name,
        tag: datasets[_selectedRunIndex].tags[_selectedTagIndex].tag,
        type: _selectedGraphType,
      }
    },
  
    _selectedRunIndexChanged: function(runIndex: number): void {
      if (!this.datasets) return;
      // Reset the states when user pick a different run.
      this.colorBy = ColorBy.STRUCTURE;
      this._selectedTagIndex = 0;
      this._selectedGraphType = this._getDefaultSelectionType();
    },
  
    _selectedTagIndexChanged(): void {
        this._selectedGraphType = this._getDefaultSelectionType();
    },
  
    _getDefaultSelectionType(): tf.graph.SelectionType {
      const {
        datasets,
        _selectedRunIndex: run,
        _selectedTagIndex: tag,
      } = this;
      if (!datasets ||
          !datasets[run] ||
          !datasets[run].tags[tag] ||
          datasets[run].tags[tag].opGraph) {
        return tf.graph.SelectionType.OP_GRAPH;
      }
      if (datasets[run].tags[tag].profile) {
        return tf.graph.SelectionType.PROFILE;
      }
      if (datasets[run].tags[tag].conceptualGraph) {
        return tf.graph.SelectionType.CONCEPTUAL_GRAPH;
      }
      return tf.graph.SelectionType.OP_GRAPH;
    },
  
  });
  
  }  // namespace tf.graph.controls
  