#!/usr/bin/env python3
"""Tests for RL memory code."""

import sys
from os.path import dirname, realpath

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.knowledge_base import SparqlEndpoint
from research.rl_environments import State, Action, Environment
from research.rl_memory import memory_architecture, NaiveDictKB, NetworkXKB, SparqlKB
from datetime import datetime


def test_memory_architecture():
    """Test the memory architecture meta-environment."""

    class TestEnv(Environment):
        """A simple environment with a single string state."""

        def __init__(self, size, index=0):
            """Initialize the TestEnv.

            Arguments:
                size (int): The length of one side of the square.
                index (int): The initial int.
            """
            super().__init__()
            self.size = size
            self.init_index = index
            self.index = self.init_index

        def get_state(self): # noqa: D102
            return State(index=self.index)

        def get_observation(self): # noqa: D102
            return State(index=self.index)

        def get_actions(self): # noqa: D102
            if self.index == -1:
                return []
            else:
                return [Action(str(i)) for i in range(-1, size * size)]

        def reset(self): # noqa: D102
            self.start_new_episode()

        def start_new_episode(self): # noqa: D102
            self.index = self.init_index

        def react(self, action): # noqa: D102
            assert action in self.get_actions()
            if action.name != 'no-op':
                self.index = int(action.name)
            if self.end_of_episode():
                return 100
            else:
                return -1

        def visualize(self): # noqa: D102
            pass

    size = 5
    env = memory_architecture(TestEnv)(
        # memory architecture
        knowledge_store=NaiveDictKB(),
        # TestEnv
        size=size,
        index=0,
    )
    env.start_new_episode()
    for i in range(size * size):
        env.add_to_ltm(index=i, row=(i // size), col=(i % size))
    # test observation
    assert env.get_observation() == State(
        perceptual_index=0,
    ), env.get_observation()
    # test actions
    assert (
        set(env.get_actions()) == set([
            *(Action(str(i)) for i in range(-1, size * size)),
            Action('copy', src_buf='perceptual', src_attr='index', dst_buf='query', dst_attr='index'),
        ])
    ), set(env.get_actions())
    # test pass-through reaction
    reward = env.react(Action('9'))
    assert env.get_observation() == State(
        perceptual_index=9,
    ), env.get_observation()
    assert reward == -1, reward
    # query test
    env.react(Action('copy', src_buf='perceptual', src_attr='index', dst_buf='query', dst_attr='index'))
    assert env.get_observation() == State(
        perceptual_index=9,
        query_index=9,
        retrieval_index=9,
        retrieval_row=1,
        retrieval_col=4,
    ), env.get_observation()
    # query with no results
    env.react(Action('copy', src_buf='retrieval', src_attr='row', dst_buf='query', dst_attr='row'))
    env.react(Action('0'))
    env.react(Action('copy', src_buf='perceptual', src_attr='index', dst_buf='query', dst_attr='index'))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_index=0,
        query_row=1,
    ), env.get_observation()
    # delete test
    env.react(Action('delete', buf='query', attr='index'))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_row=1,
        retrieval_index=5,
        retrieval_row=1,
        retrieval_col=0,
    ), env.get_observation()
    # next result test
    env.react(Action('next-result'))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_row=1,
        retrieval_index=6,
        retrieval_row=1,
        retrieval_col=1,
    ), env.get_observation()
    # delete test
    env.react(Action('prev-result'))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_row=1,
        retrieval_index=5,
        retrieval_row=1,
        retrieval_col=0,
    ), env.get_observation()
    # complete the environment
    reward = env.react(Action('-1'))
    assert env.end_of_episode()
    assert reward == 100, reward


def test_networkxkb():
    """Test the NetworkX KnowledgeStore."""
    timer = 0
    def activation_fn(graph, mem_id, activation):
        graph.nodes[mem_id]['activation'].append(activation)


    store = NetworkXKB(activation_fn=activation_fn)
    store.store('cat', is_a='mammal', has='fur', name='cat')
    store.store('bear', is_a='mammal', has='fur', name='bear')
    store.store('whale', is_a='mammal', lives_in='water')
    store.store('whale', name='whale') # this activates whale
    store.store('fish', is_a='animal', lives_in='water')
    store.store('mammal', has='vertebra', is_a='animal')
    # retrieval
    result = store.retrieve('whale')
    assert sorted(result.items()) == [('is_a', 'mammal'), ('lives_in', 'water'), ('name', 'whale')]
    # failed query
    result = store.query({'has': 'vertebra', 'lives_in': 'water'})
    assert result is None
    # unique query
    result = store.query({'has': 'vertebra'})
    assert sorted(result.items()) == [('has', 'vertebra'), ('is_a', 'animal')]
    # query traversal
    store.store('cat')
    # at this point, whale has been activated twice (from the store and the retrieve)
    # while cat has been activated once (from the store)
    # so a search for mammals will give, in order: whale, cat, bear
    result = store.query({'is_a': 'mammal'})
    assert result['name'] == 'whale'
    assert store.has_next_result
    result = store.next_result()
    print(result['name'])
    assert result['name'] == 'bear'
    assert store.has_next_result
    result = store.next_result()
    assert result['name'] == 'cat'
    assert not store.has_next_result
    assert store.has_prev_result
    result = store.prev_result()
    assert store.has_prev_result
    result = store.prev_result()
    assert result['name'] == 'whale'
    assert not store.has_prev_result
    iterator = store.graph.__iter__()
    for node in iterator:
        print(node + ":", (store.graph.nodes.get(node)['activation']))
    iterator = store.graph






def test_sparqlkb():
    """Test the SPARQL endpoint KnowledgeStore."""
    release_date_attr = '<http://dbpedia.org/ontology/releaseDate>'
    release_date_value = '"1979-11-30"^^<http://www.w3.org/2001/XMLSchema#date>'
    # connect to DBpedia
    dbpedia = SparqlEndpoint('https://dbpedia.org/sparql')
    # test retrieve
    store = SparqlKB(dbpedia)
    result = store.retrieve('<http://dbpedia.org/resource/The_Wall>')
    assert release_date_attr in result, sorted(result.keys())
    assert result[release_date_attr] == release_date_value, result[release_date_attr]
    # test query
    result = store.query({
        '<http://dbpedia.org/ontology/releaseDate>': '"1979-11-30"^^xsd:date',
        '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>': '<http://dbpedia.org/ontology/Album>',
    })
    assert release_date_attr in result, sorted(result.keys())
    assert result[release_date_attr] == release_date_value, result[release_date_attr]


def main():
    test_networkxkb()

if __name__ == '__main__':
    main()
