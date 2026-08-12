"""
Date: 2026-08-12

Covers: root_length.functions_pipeline.edit_segfiles.edit_annotation_napari —
specifically the Tools panel it builds (the `ACTIONS` table, the buttons, the
info box and the QGroupBox sections) and `edit_annotation_napari_cheekycells`.

napari's real Viewer needs an OpenGL canvas, which is not available headless, so
a stub viewer stands in and `napari.run()` is replaced by an inspection hook.
magicgui and Qt are real, which is where the panel-building code lives.

Assumed behaviour pinned down here:
- every action in `ACTIONS` gets a keybinding, whether or not it has a button
- each button caption ends in "[<key>]", so it advertises its own shortcut
- the info box lists the mouse-position actions ('r', 't') and nothing else,
  also when curr_file is None (where 'w' is buttonless but is not a mouse action)
- the panel groups widgets into the "Improve segmentation" and
  "Analysis preview" boxes, with the analysis button inside the latter
- the "Quit, no save [q]" button does what the 'q' key does
- with curr_file=None there is no save button and no reload button
- pressing 'r' routes to correct_mask_rootshootline at the cursor position
"""

import types

import numpy as np
import pytest
from qtpy.QtWidgets import (QApplication, QGroupBox, QPushButton, QWidget,
                            QLabel)

import napari

import root_length.functions_pipeline.edit_segfiles as edseg


################################################################################
# %% stub viewer: everything edit_annotation_napari touches on a napari.Viewer

@pytest.fixture(scope='session', autouse=True)
def _qapp():
    return QApplication.instance() or QApplication([])


class StubLayer:
    def __init__(self, data=None, name=''):
        self.data = data
        self.name = name
        self.translate = (0, 0)
        self.editable = True
        self.selected_label = 1
        self.brush_size = 10


class StubLayers(list):
    selection = types.SimpleNamespace(active=None)

    def __contains__(self, item):
        return any(getattr(layer, 'name', None) == item for layer in self)


class StubViewer:
    def __init__(self, title=''):
        self.window = types.SimpleNamespace(
            _qt_window=QWidget(),
            _qt_viewer=types.SimpleNamespace(
                canvas=types.SimpleNamespace(native=QWidget())),
            docked=[],
        )
        self.window.add_dock_widget = \
            lambda widget, name=None: self.window.docked.append((name, widget))
        self.layers = StubLayers()
        self.keymap = {}
        self.cursor = types.SimpleNamespace(position=(25.0, 40.0))
        self.closed = False

    def _add(self, layer):
        self.layers.append(layer)
        return layer

    def add_image(self, data, name=''):
        return self._add(StubLayer(data, name))

    def add_labels(self, data, name='', **kwargs):
        return self._add(StubLayer(data, name))

    def add_shapes(self, data, name='', **kwargs):
        return self._add(StubLayer(data, name))

    def bind_key(self, key):
        def deco(func):
            assert key not in self.keymap, f"duplicate keybinding for {key!r}"
            self.keymap[key] = func
            return func
        return deco

    def close(self):
        self.closed = True


class FakeFile:
    fullpath = '/nonexistent/fake_seg.npz'
    file_idx = 3


def _make_labels():
    """A plate with one plant: a shoot block on top of a root block."""
    seg = np.zeros((60, 80), dtype=np.uint8)
    seg[20:30, 30:50] = 1
    seg[30:40, 30:50] = 2
    return seg


def open_editor(monkeypatch, curr_file, on_open=None):
    """
    Run edit_annotation_napari against a stub viewer and hand back the result.

    `on_open` is called with the stub viewer while the "window" is open (ie from
    the patched napari.run), which is the only moment the panel and keymap can
    be exercised. Returns (viewer, panel, editor return value).
    """
    captured = {}

    def make_viewer(title=''):
        captured['viewer'] = StubViewer(title)
        return captured['viewer']

    def fake_run():
        viewer = captured['viewer']
        captured['panel'] = viewer.window.docked[0][1]
        if on_open is not None:
            on_open(viewer, captured['panel'])

    monkeypatch.setattr(napari, 'Viewer', make_viewer)
    monkeypatch.setattr(napari, 'run', fake_run)

    result = edseg.edit_annotation_napari(
        np.zeros((60, 80), dtype=np.uint8), _make_labels(),
        curr_file=curr_file, mask_rect=(5, 55, 5, 75))

    return captured['viewer'], captured['panel'], result


def buttons_of(widget):
    """Captions of every QPushButton in a Qt widget tree."""
    return [b.text() for b in widget.findChildren(QPushButton)]


def group_boxes_of(widget):
    """{title: the group box} for every QGroupBox in a Qt widget tree."""
    return {b.title(): b for b in widget.findChildren(QGroupBox)}


################################################################################
# %% the ACTIONS table

def test_every_action_gets_a_keybinding(monkeypatch):
    """All seven editor actions are bound to their key, buttons or not."""
    viewer, _, _ = open_editor(monkeypatch, FakeFile())
    assert set(viewer.keymap) == set('rtuwjnq')


def test_button_captions_advertise_their_shortcut(monkeypatch):
    """Each action button's caption ends in its own key, as "[k]"."""
    _, panel, _ = open_editor(monkeypatch, FakeFile())
    captions = buttons_of(panel)
    for key, caption in [('u', 'Relabel by lines'), ('w', 'Save now'),
                         ('j', 'Jump to sample'), ('n', 'Next file, no save'),
                         ('q', 'Quit, no save')]:
        assert f"{caption} [{key}]" in captions


def test_info_box_lists_only_the_mouse_actions(monkeypatch):
    """The info box advertises r and t, the two actions that have no button."""
    _, panel, _ = open_editor(monkeypatch, FakeFile())
    info = next(lbl.text() for lbl in panel.findChildren(QLabel)
                if 'Hover over the plant' in lbl.text())
    assert 'draw root/shoot line' in info
    assert 'draw through-line' in info
    assert 'save now' not in info.lower()


def test_info_box_excludes_buttonless_w_without_a_file(monkeypatch):
    """With curr_file=None, 'w' loses its button but is not a mouse action."""
    _, panel, _ = open_editor(monkeypatch, None)
    info = next(lbl.text() for lbl in panel.findChildren(QLabel)
                if 'Hover over the plant' in lbl.text())
    assert 'save now' not in info.lower()


################################################################################
# %% the panel layout

def test_panel_groups_widgets_into_titled_boxes(monkeypatch):
    """The size filters and the analysis settings sit in their own QGroupBox."""
    _, panel, _ = open_editor(monkeypatch, FakeFile())
    boxes = group_boxes_of(panel)
    assert set(boxes) == {'Improve segmentation', 'Analysis preview'}

    seg_buttons = buttons_of(boxes['Improve segmentation'])
    assert 'Remove smaller' in seg_buttons
    assert 'Remove larger' in seg_buttons
    assert 'Relabel by lines [u]' in seg_buttons

    assert 'Preview analysis result' in buttons_of(boxes['Analysis preview'])


def test_no_save_buttons_without_a_file(monkeypatch):
    """curr_file=None means nowhere to save, so those buttons are left out."""
    _, panel, _ = open_editor(monkeypatch, None)
    captions = buttons_of(panel)
    assert 'Save now [w]' not in captions
    assert not any('reload' in c.lower() for c in captions)
    # the rest of the panel is still there
    assert set(group_boxes_of(panel)) == {'Improve segmentation',
                                          'Analysis preview'}


################################################################################
# %% actions actually do something

def test_quit_button_matches_the_q_key(monkeypatch):
    """Clicking "Quit, no save [q]" closes the viewer and asks for the loop to stop."""
    def click_quit(viewer, panel):
        button = next(b for b in panel.findChildren(QPushButton)
                      if b.text().startswith('Quit'))
        button.click()

    viewer, _, (seg_data, _, requests) = open_editor(
        monkeypatch, FakeFile(), on_open=click_quit)

    assert viewer.closed
    assert requests['quitloop_flag'] is True
    assert seg_data is None, "quitting must not hand back labels to save"


def test_r_key_draws_at_the_cursor(monkeypatch):
    """'r' routes to correct_mask_rootshootline at the current cursor position."""
    seen = {}

    def fake_correct(mask, row, col):
        seen['rowcol'] = (row, col)
        return mask

    monkeypatch.setattr(edseg, 'correct_mask_rootshootline', fake_correct)

    def press_r(viewer, panel):
        viewer.keymap['r'](viewer)

    open_editor(monkeypatch, FakeFile(), on_open=press_r)

    # cursor is at (25, 40) and the shift widget starts at zero offset
    assert seen['rowcol'] == (25, 40)
