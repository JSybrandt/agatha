import agatha.topic_query.path_util as util
import networkx as nx

def node_sets_equal(g1:nx.Graph, g2:nx.Graph)->bool:
  a1 = {str(n): dict(a) for n, a in g1.nodes.items()}
  a2 = {str(n): dict(a) for n, a in g2.nodes.items()}
  print(a1)
  print("---")
  print(a2)
  return a1 == a2

def test_delete_attribute():
  expected = nx.Graph()
  expected.add_node("A")
  expected.add_node("B")
  expected.add_node("C", dont_delete_me=True)

  actual = nx.Graph()
  actual.add_node("A", delete_me=True)
  actual.add_node("B", delete_me=False)
  actual.add_node("C", dont_delete_me=True)
  util.clear_node_attribute(
      graph=actual,
      attribute="delete_me"
  )
  assert node_sets_equal(actual, expected)

def test_replace_attribute():
  expected = nx.Graph()
  expected.add_node("A", change_me=0)
  expected.add_node("B", change_me=0)
  # When you set reinit value, set all nodes
  expected.add_node("C", dont_change_me=3, change_me=0)

  actual = nx.Graph()
  actual.add_node("A", change_me=1)
  actual.add_node("B", change_me=2)
  actual.add_node("C", dont_change_me=3)
  util.clear_node_attribute(
      graph=actual,
      attribute="change_me",
      reinitialize=0
  )
  assert node_sets_equal(actual, expected)
