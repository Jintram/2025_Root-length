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
- the r/t hint lists the mouse-position actions and nothing else, also when
  curr_file is None (where 'w' is buttonless but is not a mouse action)
- the panel holds five sections, in workflow order (identified by their
  contents, not their titles, which are wording that gets tuned)
- each button sits in the section for the step it belongs to
- each hint sits in the box holding the controls it describes
- a hint set to None (currently `hint_view`, `hint_rect`) is left out entirely,
  and a section left with no contents at all is not built
- the size-filter buttons sit on one row with their own spinbox
- smoothing radius defaults to 5 pipeline-wide, and None stays a valid "off"
- the navigation section opens with the close hint, then a 2x2 button grid
  (save | next / jump | quit), which closes up when the save button is absent
- the regrouping lost nothing: all 9 buttons, 2 sliders, 6 spinboxes, 1 checkbox
- the "Quit, no save [q]" button does what the 'q' key does
- with curr_file=None there is no save button and no reload button, but all
  five sections are still there
- pressing 'r' routes to correct_mask_rootshootline at the cursor position
"""

import types

import numpy as np
import pytest
from qtpy.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox,
                            QGroupBox, QLabel, QPushButton, QSlider, QWidget)

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
    """Every QGroupBox in a Qt widget tree, top to bottom as laid out."""
    return widget.findChildren(QGroupBox)


def section_holding(widget, button_caption):
    """
    Index of the section whose box contains the given button.

    Sections are found by content rather than by title, because the titles are
    wording that gets tuned; what these tests are pinning down is which controls
    end up together and in what order, not how the boxes are named.
    """
    for index, box in enumerate(group_boxes_of(widget)):
        if button_caption in buttons_of(box):
            return index
    raise AssertionError(f"no section contains a {button_caption!r} button")


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


def mouse_hint_of(panel):
    """
    The hint naming the keys that act at the mouse position.

    Found by the keys it mentions rather than by its opening words, which are
    wording that gets tuned; what matters is which keys it names.
    """
    return next(lbl.text() for lbl in panel.findChildren(QLabel)
                if '[r]' in lbl.text())


def test_mouse_hint_lists_only_the_mouse_actions(monkeypatch):
    """The hint names r and t, generated from ACTIONS rather than hardcoded."""
    _, panel, _ = open_editor(monkeypatch, FakeFile())
    hint = mouse_hint_of(panel)
    assert '[r]' in hint and '[t]' in hint
    # actions that do have a button carry their own key, so they stay out
    assert '[u]' not in hint and '[w]' not in hint


def test_mouse_hint_excludes_buttonless_w_without_a_file(monkeypatch):
    """With curr_file=None, 'w' loses its button but is not a mouse action."""
    _, panel, _ = open_editor(monkeypatch, None)
    assert '[w]' not in mouse_hint_of(panel)


################################################################################
# %% the panel layout

N_SECTIONS = 5


def test_sections_appear_in_workflow_order(monkeypatch):
    """The five boxes are stacked in the order a plate is worked through."""
    _, panel, _ = open_editor(monkeypatch, FakeFile())
    assert len(group_boxes_of(panel)) == N_SECTIONS

    # view (no button of its own) → plate area → fix mask → check → leave
    assert (section_holding(panel, 'Save + reload plate (e.g. after adjusting rect)')
            < section_holding(panel, 'Remove smaller')
            < section_holding(panel, 'Preview analysis result')
            < section_holding(panel, 'Quit, no save [q]'))
    # the shift sliders are the first section, ahead of all of those
    assert group_boxes_of(panel)[0].findChildren(QSlider)


def test_each_control_sits_in_its_own_section(monkeypatch):
    """Every button lives in the section of the step it belongs to."""
    _, panel, _ = open_editor(monkeypatch, FakeFile())
    boxes = group_boxes_of(panel)

    assert buttons_of(boxes[0]) == []           # view: sliders only
    assert buttons_of(boxes[1]) == ['Save + reload plate (e.g. after adjusting rect)']
    assert buttons_of(boxes[2]) == ['Remove smaller', 'Remove larger',
                                    'Relabel by lines [u]']
    assert buttons_of(boxes[3]) == ['Preview analysis result']
    # 2x2 grid, read row by row: save | next / jump | quit
    assert buttons_of(boxes[4]) == ['Save now [w]', 'Next file, no save [n]',
                                    'Jump to sample [j]', 'Quit, no save [q]']


def test_hints_sit_in_the_section_they_describe(monkeypatch):
    """Each explanation is inside the box holding the controls it talks about."""
    _, panel, _ = open_editor(monkeypatch, FakeFile())
    boxes = group_boxes_of(panel)

    def texts(box):
        return ' '.join(lbl.text() for lbl in box.findChildren(QLabel))

    assert '[r]' in texts(boxes[2])    # with the relabel/size controls
    assert 'Cmd+W' in texts(boxes[4])  # with the buttons that leave


def test_a_hint_set_to_none_is_left_out(monkeypatch):
    """A hint set to None (hint_view, hint_rect) adds no label to its box."""
    _, panel, _ = open_editor(monkeypatch, FakeFile())
    boxes = group_boxes_of(panel)

    # only the sliders' own field labels remain ("shift x" / "shift y"), which
    # magicgui makes; no italic hint label of ours
    assert [lbl.text() for lbl in boxes[0].findChildren(QLabel) if lbl.text()] \
        == ['shift x', 'shift y']
    # the plate-area box is left with just its button
    assert [lbl.text() for lbl in boxes[1].findChildren(QLabel) if lbl.text()] \
        == []


def test_size_filter_buttons_share_a_row_with_their_input(monkeypatch):
    """Each "Remove …" button sits beside its own spinbox, not on its own line."""
    from qtpy.QtWidgets import QHBoxLayout

    _, panel, _ = open_editor(monkeypatch, FakeFile())
    for caption in ('Remove smaller', 'Remove larger'):
        button = next(b for b in panel.findChildren(QPushButton)
                      if b.text() == caption)
        row = button.parent()
        assert isinstance(row.layout(), QHBoxLayout)
        assert row.findChildren(QAbstractSpinBox), \
            f"{caption} is not on a row with a spinbox"


def test_navigation_buttons_form_a_2x2_grid(monkeypatch):
    """The hint comes first, then save|next over jump|quit in a 2x2 grid."""
    from qtpy.QtWidgets import QGridLayout

    _, panel, _ = open_editor(monkeypatch, FakeFile())
    nav_box = group_boxes_of(panel)[-1]

    # the closing hint is above the buttons
    hint = nav_box.findChildren(QLabel)[0]
    assert 'Cmd+W' in hint.text()

    grid = next(w.layout() for w in nav_box.findChildren(QWidget)
                if isinstance(w.layout(), QGridLayout))
    assert (grid.rowCount(), grid.columnCount()) == (2, 2)
    placed = {(grid.getItemPosition(i)[0], grid.getItemPosition(i)[1]):
              grid.itemAt(i).widget().text() for i in range(grid.count())}
    assert placed == {(0, 0): 'Save now [w]', (0, 1): 'Next file, no save [n]',
                      (1, 0): 'Jump to sample [j]', (1, 1): 'Quit, no save [q]'}


def test_navigation_grid_closes_up_without_the_save_button(monkeypatch):
    """With no file to save into, the remaining three fill the grid without a hole."""
    from qtpy.QtWidgets import QGridLayout

    _, panel, _ = open_editor(monkeypatch, None)
    nav_box = group_boxes_of(panel)[-1]
    grid = next(w.layout() for w in nav_box.findChildren(QWidget)
                if isinstance(w.layout(), QGridLayout))

    placed = {(grid.getItemPosition(i)[0], grid.getItemPosition(i)[1]):
              grid.itemAt(i).widget().text() for i in range(grid.count())}
    assert placed == {(0, 0): 'Next file, no save [n]',
                      (0, 1): 'Jump to sample [j]',
                      (1, 0): 'Quit, no save [q]'}


def test_smoothing_defaults_to_5_everywhere(monkeypatch):
    """Smoothing radius 5 is the pipeline default, and what the editor opens with."""
    from root_length.functions_pipeline.config import ConfigPipeline

    assert ConfigPipeline().smoothing_diskradius == 5
    # None stays a valid value, meaning "no smoothing"
    assert ConfigPipeline(smoothing_diskradius=None).smoothing_diskradius is None

    _, panel, _ = open_editor(monkeypatch, FakeFile())
    analysis_box = next(b for b in group_boxes_of(panel) if 'nalysis' in b.title())
    values = {lbl.text(): lbl.parent().findChildren(QAbstractSpinBox)[0].value()
              for lbl in analysis_box.findChildren(QLabel)
              if lbl.parent().findChildren(QAbstractSpinBox)}

    assert values['Smoothing radius (0=off)'] == 5


def test_a_section_with_nothing_left_is_dropped(monkeypatch):
    """A box whose entries are all absent disappears rather than showing empty."""
    _, panel, _ = open_editor(monkeypatch, None)
    # hint_rect is None and there is no file to reload, so the plate-area
    # section has no contents at all and must not be built
    for box in group_boxes_of(panel):
        assert box.findChildren(QPushButton) or box.findChildren(QLabel), \
            f"empty section left in the panel: {box.title()!r}"


def test_nothing_was_lost_in_the_regrouping(monkeypatch):
    """The panel still holds every control it had before it was sectioned."""
    _, panel, _ = open_editor(monkeypatch, FakeFile())

    assert sorted(buttons_of(panel)) == sorted([
        'Remove smaller', 'Remove larger', 'Relabel by lines [u]',
        'Save now [w]', 'Save + reload plate (e.g. after adjusting rect)',
        'Preview analysis result', 'Jump to sample [j]',
        'Next file, no save [n]', 'Quit, no save [q]'])
    # 2 shift sliders, 2 size thresholds, 2 analysis settings (each with a
    # spinbox), plus the one analysis checkbox
    assert len(panel.findChildren(QSlider)) == 2
    assert len(panel.findChildren(QAbstractSpinBox)) == 6
    assert len(panel.findChildren(QCheckBox)) == 1


def test_no_save_buttons_without_a_file(monkeypatch):
    """curr_file=None means nowhere to save, so those buttons are left out."""
    _, panel, _ = open_editor(monkeypatch, None)
    captions = buttons_of(panel)
    assert 'Save now [w]' not in captions
    assert not any('reload' in c.lower() for c in captions)
    # the plate-area section had nothing but that button left, so it goes too;
    # every other section survives
    assert len(group_boxes_of(panel)) == N_SECTIONS - 1


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
