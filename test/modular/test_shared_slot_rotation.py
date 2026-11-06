"""Nodes that share a resource never hold one double-buffer slot back to back."""
from types import SimpleNamespace

from mstar.engine.engine import SlotRotation, SubmoduleManagement, share_slot_rotations


def _node(*resources, num_slots=2):
    node = object.__new__(SubmoduleManagement)
    node.resources = {str(i): resource for i, resource in enumerate(resources)}
    node._slots = SlotRotation()
    node._slots.num_slots = num_slots
    return node


def test_nodes_sharing_a_resource_alternate_slots():
    shared = SimpleNamespace()
    prefix, action = _node(shared), _node(shared)
    share_slot_rotations([prefix, action])
    assert [prefix.lease_slot(), action.lease_slot(), prefix.lease_slot(), action.lease_slot()] == [0, 1, 0, 1]
    assert action.next_slot == prefix.next_slot == 0


def test_nodes_sharing_nothing_rotate_apart():
    vit, llm = _node(SimpleNamespace()), _node(SimpleNamespace())
    share_slot_rotations([vit, llm])
    assert [vit.lease_slot(), llm.lease_slot()] == [0, 0]


def test_a_chain_of_shared_resources_is_one_rotation():
    a, b = SimpleNamespace(), SimpleNamespace()
    first, middle, last = _node(a), _node(a, b), _node(b)
    other_order = [last, first, middle]
    share_slot_rotations(other_order)
    assert first._slots is middle._slots is last._slots


def test_the_rotation_covers_the_most_slots_any_sharer_needs():
    shared = SimpleNamespace()
    one, two = _node(shared, num_slots=1), _node(shared, num_slots=2)
    share_slot_rotations([one, two])
    assert one.num_slots == two.num_slots == 2


def test_a_later_merge_reaches_every_node_of_the_merged_rotation():
    a, b = SimpleNamespace(), SimpleNamespace()
    first, second, other, bridge = _node(a), _node(a), _node(b), _node(a, b)
    share_slot_rotations([first, second, other, bridge])
    assert first._slots is second._slots is other._slots is bridge._slots

