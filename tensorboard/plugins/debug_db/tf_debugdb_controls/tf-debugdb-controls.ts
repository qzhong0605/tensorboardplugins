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
namespace tf.debug.controls {

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
    type: tf.debug.SelectionType,
  }
  
  export interface DeviceForStats {
    [key: string]: boolean;
  }
  
  // TODO(stephanwlee): Move this to tf-debugdb-dashboard
  export interface TagItem {
    tag: string | null,
    displayName: string,
    conceptualGraph: boolean,
    opGraph: boolean,
    profile: boolean,
  }
  
  // TODO(stephanwlee): Move this to tf-debugdb-dashboard
  export interface RunItem {
    name: string,
    tags: TagItem[],
  }
  
  // TODO(stephanwlee): Move this to tf-debugdb-dashboard
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
  
  // TODO(stephanwlee) Move this to tf-debugdb.html when it becomes TypeScript.
  interface ColorByParams {
    compute_time: ColorParams,
    memory: ColorParams,
    device: DeviceColor[],
    xla_cluster: XlaClusterColor[],
  }
  
  const GRADIENT_COMPATIBLE_COLOR_BY: Set<ColorBy> = new Set([
      ColorBy.COMPUTE_TIME, ColorBy.MEMORY]);
  
  
  Polymer({
    is: 'tf-debugdb-controls',
    properties: {
      addIdentification: {
        type: Object,
        notify: true,
      },
      selectedMachine:{
        type: Number,
        observer:'_selectedMachineChanged',
      },
      selectedId:{
        type: Number,
        observer:'_selectedIdChanged'
      },
      selectedMachineList:{
        type: Array,
        value: [],
      },
      idList:{
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
        value: 0,
        observer:'_modeChanged'
      },
      _srcTypes:{
        type: Array,
        value: ['onnx', 'caffe', 'caffe2', 'torch', 'tf'],
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
       * @type {?tf.debug.proto.StepStats}
       */
      stats: {
        value: null,
        type: Object,
        // observer: '_statsChanged',
      },
      
      /**
       * @type {!tf.debug.controls.ColorBy}
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
      },
      attachParam: {
        type: Object,
        notify: true,
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
       * @type {tf.debug.SelectionType}
       */
      _selectedGraphType: {
        type: String,
        value: tf.debug.SelectionType.OP_GRAPH,
      },
    },
  
    listeners: {},
  
    ready: function(){},
    _modeChanged: function(){
      if(document.getElementById('c2')){
        if(this._srcMode == 2){//c2
          document.getElementById('elsefile').style.display = ''
          document.getElementById('elsepath').style.display = 'none'
          document.getElementById('torch').style.display = 'none'
          document.getElementById('c2').style.display = ''
          return 
        }
        if(this._srcMode == 3){//torch
          document.getElementById('elsefile').style.display = 'none'
          document.getElementById('elsepath').style.display = ''
          document.getElementById('torch').style.display = ''
          document.getElementById('c2').style.display = 'none'
          return 
        }
        document.getElementById('elsefile').style.display = ''
        document.getElementById('elsepath').style.display = ''
        document.getElementById('torch').style.display = 'none'
        document.getElementById('c2').style.display = 'none'
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
      var mthis = this
      var attachList = []
      this.attachList.forEach(element => {
        attachList.push(element);
      });
      var addIdentification = {}
      var loadparams = {}
      const params = new URLSearchParams();
      var model_type = this._srcTypes[this._srcMode];
      params.set('model_type', model_type);
      loadparams['model_type'] = model_type;

      if(this._srcMode == 2){
        var file_type = (<HTMLInputElement>document.getElementById('fileType')).value;
        var predict_net = (<HTMLInputElement>document.getElementById('predictNet')).value;
        var init_net = (<HTMLInputElement>document.getElementById('initNet')).value;
        params.set('predict_net', predict_net);
        params.set('init_net', init_net);
        params.set('file_type', file_type);
        loadparams['predict_net'] = predict_net
        loadparams['init_net'] = init_net
        loadparams['file_type'] = file_type
      }
      else{
        if(this._srcMode == 3){
          var inputTensorSize = (<HTMLInputElement>document.getElementById('inputTensorSize')).value;
          var source_path = (<HTMLInputElement>document.getElementById('srcPath')).value;
          params.set('source_path', source_path);
          params.set('input_tensor_size', inputTensorSize);
          loadparams['source_path'] = source_path
          loadparams['input_tensor_size'] = inputTensorSize
        }
        else{
          var file_type = (<HTMLInputElement>document.getElementById('fileType')).value;
          var source_path = (<HTMLInputElement>document.getElementById('srcPath')).value;
          params.set('source_path', source_path);
          params.set('file_type', file_type);
          loadparams['source_path'] = source_path
          loadparams['file_type'] = file_type
        }
      }
      // this.selection = loadparams

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
        fetch(tf_backend.getRouter().pluginRoute('debugdb', '/newstart', params)).then((res) => {
          // Fetch does not reject for 400+.
          if(res.ok){
            res.json().then(function(respond){
              addIdentification = {
                identification: respond.identification,
                model_type: model_type,
                iteration: totalIteration,
                memory_size: memorySize,
                learning_rate: learningRate,
                optimization_method: optimizationMethod,
              }
              mthis.addIdentification = addIdentification;
              attachList.push(respond.identification)
              mthis.attachList = attachList;
              mthis.attachMap[respond.identification] = respond.machineList;
              
              (<HTMLInputElement>document.getElementById('fileType')).value = '';
              (<HTMLInputElement>document.getElementById('srcPath')).value = '';
              (<HTMLInputElement>document.getElementById('predictNet')).value = '';
              (<HTMLInputElement>document.getElementById('initNet')).value = '';
              (<HTMLInputElement>document.getElementById('batchSize')).value = '';
              (<HTMLInputElement>document.getElementById('memorySize')).value = '';
              (<HTMLInputElement>document.getElementById('optimizationMethod')).value = '';
              (<HTMLInputElement>document.getElementById('learningRate')).value = '';
              (<HTMLInputElement>document.getElementById('totalIteration')).value = '';
              (<HTMLInputElement>document.getElementById('inputTensorSize')).value = '';
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
        fetch(tf_backend.getRouter().pluginRoute('debugdb', '/newstop', params)).then((res) => {
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
        fetch(tf_backend.getRouter().pluginRoute('debugdb', '/newcontinue', params)).then((res) => {
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
      // this.attachParam = network_identification;
      const params = new URLSearchParams();
      params.set('network_identification',network_identification);
      new Promise(() => {
        fetch(tf_backend.getRouter().pluginRoute('debugdb', '/attach', params)).then((res) => {
          // Fetch does not reject for 400+.
          if(res.ok){
            res.json().then(function(respond){
              attachList.push(network_identification);
              mthis.attachList = attachList;
              mthis.attachMap[network_identification] = respond.attach;
              (<HTMLInputElement>document.getElementById('network_identification')).value = ''
              mthis.selectedIdentification = attachList.length-1;
              mthis.addIdentification = respond.session
            })
          }
        });
      });
    },
    _selectedIdentificationChanged: function(){
      document.getElementById('attachInfo').style.display = '';
      var selectedIdentification = this.attachList[this.selectedIdentification];
      // this.attachParam = selectedIdentification;
      var attachInfo = this.attachMap[selectedIdentification];
      (<HTMLInputElement>document.getElementById('model_type')).value = attachInfo.model_type;
      var machineList = [];
      attachInfo.list.forEach(item => {
        if(machineList.indexOf(item.m)==-1){
          machineList.push(item.m);
        }
      })
      this.selectedMachineList = machineList;
      this._selectedMachineChanged()
    },
    attachStop: function(){
      const params = new URLSearchParams();
      params.set('identification',this.attachList[this.selectedIdentification]);
      new Promise(() => {
        fetch(tf_backend.getRouter().pluginRoute('debugdb', '/attachstop', params)).then((res) => {
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
        fetch(tf_backend.getRouter().pluginRoute('debugdb', '/attachcontinue', params)).then((res) => {
          // Fetch does not reject for 400+.
          if(res.ok){
            res.text().then(function(msg){
              
            })
          }
        });
      });
    },
    _selectedMachineChanged: function(){
      var machine = this.selectedMachineList[this.selectedMachine]
      var selectedIdentification = this.attachList[this.selectedIdentification];
      var attachInfo = this.attachMap[selectedIdentification];
      var idList = [];
      attachInfo.list.forEach(item => {
        if(item.m == machine){
          idList.push(item.id);
        }
      })
      this.idList = idList
      this._selectedIdChanged()
    },
    _selectedIdChanged: function(){
      var id = this.idList[this.selectedId]
      var machine = this.selectedMachineList[this.selectedMachine]
      var selectedIdentification = this.attachList[this.selectedIdentification];
      var attachInfo = this.attachMap[selectedIdentification];
      
      attachInfo.list.forEach(item => {
        if(item.m == machine && item.id == id){
          (<HTMLInputElement>document.getElementById('batch_size')).value = item.batch_size;
          (<HTMLInputElement>document.getElementById('memory_size')).value = item.memory_size;
        }
      })
    },
  
    _datasetsChanged: function(newDatasets: Dataset, oldDatasets: Dataset) {
      if (oldDatasets != null) {
        // Select the first dataset by default.
        this._selectedRunIndex = 0;
      }
    },
  
    _computeSelection: function(datasets: Dataset, _selectedRunIndex: number,
        _selectedTagIndex: number, _selectedGraphType: tf.debug.SelectionType) {
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
  
    _getDefaultSelectionType(): tf.debug.SelectionType {
      const {
        datasets,
        _selectedRunIndex: run,
        _selectedTagIndex: tag,
      } = this;
      if (!datasets ||
          !datasets[run] ||
          !datasets[run].tags[tag] ||
          datasets[run].tags[tag].opGraph) {
        return tf.debug.SelectionType.OP_GRAPH;
      }
      if (datasets[run].tags[tag].profile) {
        return tf.debug.SelectionType.PROFILE;
      }
      if (datasets[run].tags[tag].conceptualGraph) {
        return tf.debug.SelectionType.CONCEPTUAL_GRAPH;
      }
      return tf.debug.SelectionType.OP_GRAPH;
    },
  
  });
  
  }  // namespace tf.debug.controls
  