/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
module tf.convert {

describe('graph', () => {
  let assert = chai.assert;

  it('graphlib exists', () => { assert.isTrue(graphlib != null); });

  it('simple graph contruction', () => {
    let pbtxt = tf.convert.test.util.stringToArrayBuffer(`
      node {
        name: "Q"
        op: "Input"
      }
      node {
        name: "W"
        op: "Input"
      }
      node {
        name: "X"
        op: "MatMul"
        input: "Q:2"
        input: "W"
      }`);
    let statsPbtxt = tf.convert.test.util.stringToArrayBuffer(`step_stats {
      dev_stats {
        device: "cpu"
        node_stats {
          node_name: "Q"
          all_start_micros: 10
          all_end_rel_micros: 4
        }
        node_stats {
          node_name: "Q"
          all_start_micros: 12
          all_end_rel_micros: 4
        }
      }
    }`);

    let buildParams: tf.convert.BuildParams = {
      enableEmbedding: true,
      inEmbeddingTypes: ['Const'],
      outEmbeddingTypes: ['^[a-zA-Z]+Summary$'],
      refEdges: {}
    };
    let dummyTracker =
        tf.convert.util.getTracker({set: () => { return; }, progress: 0});
    let slimGraph: SlimGraph;
    return tf.convert.parser.parseGraphPbTxt(pbtxt)
        .then(nodes => tf.convert.build(nodes, buildParams, dummyTracker))
        .then((graph: SlimGraph) => slimGraph = graph)
        .then(() => {
          assert.isTrue(slimGraph.nodes['X'] != null);
          assert.isTrue(slimGraph.nodes['W'] != null);
          assert.isTrue(slimGraph.nodes['Q'] != null);

          let firstInputOfX = slimGraph.nodes['X'].inputs[0];
          assert.equal(firstInputOfX.name, 'Q');
          assert.equal(firstInputOfX.outputTensorKey, '2');

          let secondInputOfX = slimGraph.nodes['X'].inputs[1];
          assert.equal(secondInputOfX.name, 'W');
          assert.equal(secondInputOfX.outputTensorKey, '0');
        })
        .then(() => tf.convert.parser.parseStatsPbTxt(statsPbtxt))
        .then(stepStats => {
          tf.convert.joinStatsInfoWithGraph(slimGraph, stepStats);
          assert.equal(slimGraph.nodes['Q'].stats.getTotalMicros(), 6);
        });
  });

  it('health pill numbers round correctly', () => {
    // Integers are rounded to the ones place.
    assert.equal(tf.convert.scene.humanizeHealthPillStat(42.0, true), '42');

    // Numbers with magnitude >= 1 are rounded to the tenths place.
    assert.equal(tf.convert.scene.humanizeHealthPillStat(1, false), '1.0');
    assert.equal(tf.convert.scene.humanizeHealthPillStat(42.42, false), '42.4');
    assert.equal(tf.convert.scene.humanizeHealthPillStat(-42.42, false), '-42.4');

    // Numbers with magnitude < 1 are written in scientific notation rounded to
    // the tenths place.
    assert.equal(tf.convert.scene.humanizeHealthPillStat(0, false), '0.0e+0');
    assert.equal(tf.convert.scene.humanizeHealthPillStat(0.42, false), '4.2e-1');
    assert.equal(
        tf.convert.scene.humanizeHealthPillStat(-0.042, false), '-4.2e-2');
  });

  // TODO: write tests.
});

}  // module tf.convert
