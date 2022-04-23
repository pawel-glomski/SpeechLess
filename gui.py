import imgui
import pyglet
from imgui.integrations.pyglet import create_renderer

from testwindow import show_test_window
from avsource import SLSource


def update(dt):
  imgui.new_frame()
  if imgui.begin_main_menu_bar():
    if imgui.begin_menu('File', True):
      clicked_quit, selected_quit = imgui.menu_item('Quit', 'Cmd+Q', False, True)
      if clicked_quit:
        exit()
      imgui.end_menu()
    imgui.end_main_menu_bar()


def main():
  window = pyglet.window.Window(width=1920, height=1080, resizable=True)

  imgui.create_context()
  impl = create_renderer(window)

  src = SLSource('yyy/wyk_small.mp4')

  video_player = pyglet.media.Player()
  # manually create a square texture - this is slightly modified version of Player._create_texture
  # function, which creates a rectangle texture instead of a square one
  video_player._texture = pyglet.image.Texture.create(src.video_format.width,
                                                      src.video_format.height,
                                                      rectangle=False)
  video_player._texture = video_player._texture.get_transform(flip_y=True)
  video_player._texture.anchor_y = 0

  video_player.queue(src)
  video_player.play()

  def draw(dt):
    update(dt)
    window.clear()

    # show_test_window()
    imgui.begin('test')

    imgui.text('this is a test')
    video_texture = video_player.texture
    imgui.image(texture_id=video_texture.id,
                width=video_texture.width,
                height=video_texture.height,
                uv0=(0, 0),
                uv1=((video_texture.width / video_texture.owner.width),
                     (video_texture.height / video_texture.owner.height)))

    imgui.end()

    imgui.render()
    impl.render(imgui.get_draw_data())

  pyglet.clock.schedule_interval(draw, 1 / 60)
  pyglet.app.run()
  video_player.pause()
  impl.shutdown()


if __name__ == '__main__':
  main()
