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
namespace tf.graph.edit.loader {

interface GraphRunTag {
  run: string;
  tag?: string;
}

Polymer({
  is: 'tf-graph-edit-dashboard-loader',

  properties: {
    loadparams:{
      type: Object,
      value:{},
    },
    datasets: Array,
    /**
     * @type {{value: number, msg: string}}
     *
     * A number between 0 and 100 denoting the % of progress
     * for the progress bar and the displayed message.
     */
    progress: {
      type: Object,
      notify: true,
    },
    selection: Object,
    /**
     * TODO(stephanwlee): This should be changed to take in FileList or
     * the prop should be changed to `fileInput`.
     * @type {?Event}
     */
    selectedFile: Object,
    compatibilityProvider: {
      type: Object,
      value: () => new tf.graph.edit.op.TpuCompatibilityProvider(),
    },
    hierarchyParams: {
      type: Object,
      value: () => tf.graph.edit.hierarchy.DefaultHierarchyParams,
    },
    outGraphHierarchy: {
      type: Object,
      // readOnly: true, //readonly so outsider can't change this via binding
      notify: true
    },
    reGraphHierarchy: {
      type: Object,
      // readOnly: true, //readonly so outsider can't change this via binding
      notify: true
    },
    outGraph: {
      type: Object,
      // readOnly: true, //readonly so outsider can't change this via binding
      notify: true
    },
    /** @type {Object} */
    outStats: {
      type: Object,
      readOnly: true, // This property produces data.
      notify: true
    },
    /**
     * @type {?GraphRunTag}
     */
    _graphRunTag: Object,
    datasource: Number,
  },
  observers: [
    '_selectionChanged(selection, compatibilityProvider)',
    '_selectedFileChanged(selectedFile, compatibilityProvider)',
  ],
  _selectionChanged(): void {
    // selection can change a lot within a microtask.
    // Don't fetch too much too fast and introduce race condition.
    if(this.datasource == 2){
      this.debounce('selectionchange', () => {
        // TODO:
        this._load(this.loadparams);
      });
    }
    if(this.datasource == 1){
      new Promise(() => {
        fetch(tf_backend.getRouter().pluginRoute('graphedit', '/init')).then((res) => {
          // Fetch does not reject for 400+.
          if(res.ok){
            res.text().then(function(msg){
              console.info(msg)
            })
          }
        });
      });

      this.outGraph = {
        nodes:{},
        edges:[],
      }
      this.set('progress', {
        value: 100,
        msg: "Namespace hierarchy: Finding similar subgraphs",
      });
      this._ReConstructHierarchicalGraph()
    }
  },
  _load: function(loadparams): Promise<void> {
    // Clear stats about the previous graph.
    this._setOutStats(null);
    const params = new URLSearchParams();
    params.set('model_type', loadparams.modelType);
    if(loadparams.modelType == 'caffe2'){
      params.set('file_type', loadparams.fileType);
      params.set('predict_net',loadparams.predictNet)
      params.set('init_net',loadparams.initNet)
    }
    else{
      if(loadparams.modelType == 'torch'){
        params.set('input_tensor_size', loadparams.inputTensorSize);
        params.set('model_file', loadparams.srcPath);
      }
      else{
        params.set('file_type', loadparams.fileType);
        params.set('model_file', loadparams.srcPath);
      }
    }
    
    const graphPath =
        tf_backend.getRouter().pluginRoute('graphedit', '/load', params);
    return this._fetchAndConstructHierarchicalGraph(graphPath).then(() => {
      // console.info('yes')
    })
    return ;
      
  },

  _readAndParseMetadata: function(path: string): void {
    // Reset the progress bar to 0.
    this.set('progress', {
      value: 0,
      msg: '',
    });
    var tracker = tf.graph.edit.util.getTracker(this);
    tf.graph.edit.parser.fetchAndParseMetadata(path, tracker)
        .then((stats) => {
          this._setOutStats(stats);
        });
  },
  _fetchAndConstructHierarchicalGraph: async function(
      path: (string|null), pbTxtFile?: Blob): Promise<void> {
    // Reset the progress bar to 0.
    this.set('progress', {
      value: 0,
      msg: '',
    });
    const tracker = tf.graph.edit.util.getTracker(this);
    return tf.graph.edit.loader.fetchAndConstructHierarchicalGraph(
      tracker,
      path,
      pbTxtFile,
      this.compatibilityProvider,
      this.hierarchyParams,
    ).then(({graph, graphHierarchy}) => {
      // this._setOutGraph(graph);
      // this._setOutGraphHierarchy(graphHierarchy);
      this.outGraph = graph
      this.outGraphHierarchy = graphHierarchy
    });
  },

  _ReConstructHierarchicalGraph: async function(){
    var mthis = this
    var graph = this.outGraph
    const tracker = tf.graph.edit.util.getTracker(this)
    tf.graph.edit.loader.ReConstructHierarchicalGraph(
      graph,
      tracker,
      this.compatibilityProvider,
      this.hierarchyParams,
    ).then(function(graphHierarchy){
      // this.outGraphHierarchy = graphHierarchy
      mthis.outGraphHierarchy = graphHierarchy
    })
  },

  _selectedFileChanged: function(e: Event|null): void {
    if (!e) {
      return;
    }
    const target = (e.target as HTMLInputElement);
    const file = target.files[0];
    if (!file) {
      return;
    }

    // Clear out the value of the file chooser. This ensures that if the user
    // selects the same file, we'll re-read it.
    target.value = '';

    this._fetchAndConstructHierarchicalGraph(null, file);
  },

  _deleteEdge: function(edge){
    const params = new URLSearchParams();
    params.set('type', 'delete_edge');
    params.set('data', JSON.stringify({v:edge.v,w:edge.w}));

    new Promise(() => {
      fetch(tf_backend.getRouter().pluginRoute('graphedit', '/edit', params)).then((res) => {
        // Fetch does not reject for 400+.
        if(res.ok){
          res.text().then(function(msg){
            document.querySelector('#controls').getElementsByClassName('warning')[0].innerHTML = msg
          })
        }
      });
    });

    var pos = -1;
    for (var i=0;i<this.outGraph.edges.length;i++){
      var tedge = this.outGraph.edges[i];
      if(tedge.v == edge.v && tedge.w == edge.w){
        pos = i;
        break;
      }
    }
    if(pos!=-1){
      this.outGraph.edges.splice(pos,1);
    }
    var inputs = this.outGraph.nodes[edge.w].inputs;
    pos = -1;
    for(var i=0;i<inputs.length;i++){
      if(inputs[i].name == edge.v){
        pos = i;
      }
    }
    if(pos!=-1){
      inputs.splice(pos,1);
    }
    this._ReConstructHierarchicalGraph()
  },

  _addEdge: function(src,tar, edge_type){
    const params = new URLSearchParams();
    params.set('type', 'add_edge');
    params.set('data', JSON.stringify({v:src, w:tar, 'edge_type':edge_type}));

    new Promise(() => {
      fetch(tf_backend.getRouter().pluginRoute('graphedit', '/edit', params)).then((res) => {
        // Fetch does not reject for 400+.
        if(res.ok){
          res.text().then(function(msg){
            document.querySelector('#controls').getElementsByClassName('warning')[0].innerHTML = msg
          })
        }
      });
    });

    var edge = {
      isControlDependency: false,
      isReferenceEdge: false,
      isSiblingEdge: false,
      outputTensorKey: "0",
      v:src,
      w:tar,
    };
    edge.isReferenceEdge = edge_type==='Reference Edge'
    edge.isControlDependency = edge_type==='Control Dependency Edge'
    edge.isSiblingEdge = edge_type==='Sibling Edge'
    
    this.outGraph.edges.push(edge);
    var inputs = this.outGraph.nodes[tar].inputs;
    inputs.push({
      isReferenceEdge: edge_type==='Reference Edge',
      isControlDependency: edge_type==='Control Dependency Edge',
      isSiblingEdge: edge_type==='Sibling Edge',
      name: src,
      outputTensorKey: '0',
    })
    this._ReConstructHierarchicalGraph()
  },

  _editNodeInfo: function(node){
    const params = new URLSearchParams();
    params.set('type', 'edit');
    params.set('data', JSON.stringify(node));
   
    
    new Promise(() => {
      fetch(tf_backend.getRouter().pluginRoute('graphedit', '/edit', params)).then((res) => {
        // Fetch does not reject for 400+.
        if(res.ok){
          res.text().then(function(msg){
            document.querySelector('#controls').getElementsByClassName('warning')[0].innerHTML = msg
          })
        }
      });
    });

    var n = this.outGraph.nodes[node.name];
    n.attr = node.attr;
    var h = this.outGraphHierarchy.index[node.name]
    h.attr = node.attr
  },

  _addOp: function(n){
    const params = new URLSearchParams();
    params.set('type', 'add_node');
    params.set('data', JSON.stringify(n));
   
    
    new Promise(() => {
      fetch(tf_backend.getRouter().pluginRoute('graphedit', '/edit', params)).then((res) => {
        // Fetch does not reject for 400+.
        if(res.ok){
          res.text().then(function(msg){
            document.querySelector('#controls').getElementsByClassName('warning')[0].innerHTML = msg
          })
        }
      });
    });

    let node = new OpNodeImpl(n);
    this.outGraph.nodes[n.name] = node;
    // inputs
    this.outGraph.nodes[n.name].compatible = true;
    this._ReConstructHierarchicalGraph()
  },

  _deleteOp: function(n){
    const params = new URLSearchParams();
    params.set('type', 'delete_node');
    params.set('data', JSON.stringify({op:n}));
   
    new Promise(() => {
      fetch(tf_backend.getRouter().pluginRoute('graphedit', '/edit', params)).then((res) => {
        // Fetch does not reject for 400+.
        if(res.ok){
          res.text().then(function(msg){
            document.querySelector('#controls').getElementsByClassName('warning')[0].innerHTML = msg
          })
        }
      });
    });
      
    var list = [];
    for (var i=this.outGraph.edges.length-1;i>=0;i--){
      var edge = this.outGraph.edges[i];
      if(edge.v==n){
        this.outGraph.edges.splice(i,1);
        list.push(edge.w);
      }
      if(edge.w == n){
        this.outGraph.edges.splice(i,1);
      }
    }
    delete this.outGraph.nodes[n];
    list.forEach((d)=>{
      var input = this.outGraph.nodes[d].inputs;
      for(var i=input.length-1;i>=0;i--){
        if(input[i].name == n){
          input.splice(i,1);
        }
      }
    })
    this._ReConstructHierarchicalGraph()
  },
});

}  // namespace tf.graph.edit.loader
