"""
Author: Olav Ausland Onstad

TLDR: This Code Is Awful, Will Fix Sometime In The Future :-)
Far From Finished, I Just Do Not Have Time At The Moment
"""

from PIL import Image as Img, ImageTk
from tkinter import filedialog
from tkinter import *

import pyvirtualcam
import numpy as np
import threading
import random
import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


class Application(Tk):
    camera_config = (640, 360, 30)
    frame = np.ones((camera_config[1], camera_config[0]))
    current_image = int()
    initialize = True
    images = list()
    exit = False

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self._frame = None
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.switch_frame(MainPage)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.grid()

    def _quit(self):
        self.exit = True
        self.previewing = False
        sys.exit()
        self.destroy()

    def display(self, preview):
        with pyvirtualcam.Camera(width=self.camera_config[0], height=self.camera_config[1],
                                 fps=self.camera_config[2]) as cam:
            print(f'Using virtual camera: {cam.device}')
            frame = np.zeros((cam.height, cam.width, 3), np.uint8)
            while True:
                if self.exit:
                    break
                ret, feed = cap.read()
                feed = cv2.cvtColor(feed, cv2.COLOR_BGR2RGB)

                frame[:] = feed

                # Optimize
                face_detection(frame, self)
                for image in self.images:
                    image.animate(frame, self)

                self.frame = frame
                image = ImageTk.PhotoImage(image=Img.fromarray(frame))
                preview.configure(image=image)
                preview.image = image
                cam.send(frame)
                cam.sleep_until_next_frame()


class MainPage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.config = Button(self, text="Configure", command=lambda: master.switch_frame(ConfigurePage))
        self.add = Button(self, text="Add File", command=lambda: self.add_image())

        self.config.grid(row=0, column=0, sticky="WE")
        self.add.grid(row=0, column=1, sticky="WE")

        self.preview = Label(self, text="Video Feed", image=ImageTk.PhotoImage(image=Img.fromarray(np.ones((master.camera_config[1], master.camera_config[1])))))
        self.preview.grid(row=2, column=0, columnspan=2, sticky="NEWS")

        threading.Thread(target=master.display, args=(self.preview,)).start()

    def add_image(self):
        try:
            path = str(filedialog.askopenfilename())
            name = path.split('/')[-1]

            img = np.array(cv2.imread(path, -1))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            self.master.images.append(Image(img, name, [random.randint(1, 1),
                                                        random.randint(1, 1)]
                                            , [random.randint(-2, 2), random.randint(-2, 2)]))
            if self.master.images[-1].error:
                del self.master.images[-1]
        except Exception as error:
            print(error)


class ConfigurePage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.header = Label(self, text="Configure")
        self.back_btn = Button(self, text="Back", command=lambda: master.switch_frame(MainPage))

        self.header.grid(row=0, column=0, columnspan=12, sticky="NSEW")
        self.back_btn.grid(row=1, column=0, columnspan=12, sticky="NSEW")

        self.list_images()

    def list_images(self):
        for index, image in enumerate(self.master.images):
            image.index = index
            color = "#00FF00"
            if image.error:
                color = "#FF0000"
            Label(self, text=f"{image.index}").grid(row=3 + index, column=0, sticky="WE")

            button = Button(self, text=image.name, bg=color, command=lambda image=image: self.edit_image(image))
            if index != 0:
                upshift = Button(self, text='▲')
                upshift.grid(row=3 + index, column=2, sticky="WE")
                upshift["command"] = lambda image=image: self.upshift_image(image)
            if index != len(self.master.images) - 1:
                downshift_btn = Button(self, text='▼', command=lambda image=image: self.downshift_image(image))
                downshift_btn.grid(row=3 + index, column=3, sticky="WE")

            button.grid(row=3 + index, column=1, sticky="WE")

    def edit_image(self, image):
        self.master.current_image = image.index
        self.master.switch_frame(EditImagePage)

    def upshift_image(self, image):
        curr_index = image.index
        self.master.images[curr_index] = self.master.images[curr_index - 1]
        self.master.images[curr_index - 1] = image

        self.master.switch_frame(ConfigurePage)

    def downshift_image(self, image):

        curr_index = image.index
        self.master.images[curr_index] = self.master.images[curr_index + 1]
        self.master.images[curr_index + 1] = image

        self.master.switch_frame(ConfigurePage)


class EditImagePage(Frame):
    def __init__(self, master):
        self.images = master.images
        self.image = master.images[master.current_image]
        Frame.__init__(self, master)
        Label(self, text="Velocity").grid(column=0, row=0)
        Label(self, text="Position").grid(column=0, row=1, sticky="S")
        Label(self, text="Scale").grid(column=0, row=3, sticky="S")

        self.checked = IntVar()
        self.checked.set(self.image.auto_rotate)

        self.var_pos_x = IntVar()
        self.var_pos_x.set(self.image.position[0])

        self.var_pos_y = IntVar()
        self.var_pos_y.set(self.image.position[1])

        self.var_size_x = IntVar()
        self.var_size_y = IntVar()

        self.var_size_x.set(self.image.source.shape[0])
        self.var_size_y.set(self.image.source.shape[1])

        self.var_rotate_speed = IntVar()
        self.var_rotate_speed.set(self.image.rotate_speed)

        self.x_vel = Entry(self)
        self.x_vel.insert(END, self.image.force[0])

        self.y_vel = Entry(self)
        self.y_vel.insert(END, self.image.force[1])

        self.x_pos = Scale(self, from_=0,
                           to=self.master.camera_config[0] - self.image.source.shape[0], variable=self.var_pos_x,
                           orient=HORIZONTAL)

        self.y_pos = Scale(self, from_=0,
                           to=self.master.camera_config[1] - self.image.source.shape[1], variable=self.var_pos_y
                           , orient=HORIZONTAL)

        self.auto_rotate = Checkbutton(self, text="Auto Rotate", variable=self.checked,
                                       command=lambda: self.toggle_auto_rotate())
        self.angle = Entry(self)
        self.angle.insert(END, self.image.angle)

        self.rotate_speed = Scale(self, from_=-6, to=6, orient=HORIZONTAL, variable=self.var_rotate_speed,
                                  command=lambda x=None: self.set_rotate_speed())

        # CHANGE TO BE DYNAMIC
        self.scl_x = Scale(self, from_=32, to=master.camera_config[0], variable=self.var_size_x, orient=HORIZONTAL)
        self.scl_y = Scale(self, from_=32, to=master.camera_config[1], variable=self.var_size_y, orient=HORIZONTAL)

        self.vel_btn = Button(self, text="Change Velocity", command=lambda: self.change_velocity())
        self.pos_btn = Button(self, text="Change Position", command=lambda: self.change_position())
        self.ang_btn = Button(self, text="Set Angle", command=lambda: self.set_angle())
        self.sze_btn = Button(self, text="Change Size", command=lambda: self.set_scale())
        self.del_btn = Button(self, text="Delete", command=lambda: self.delete_image())
        self.trk_btn = Button(self, text="Track / Follow Face", command=lambda: [self.follow_face(), self.image.toggle_face_tracking()])
        self.back_btn = Button(self, text="Back", command=lambda: master.switch_frame(ConfigurePage))

        self.x_vel.grid(row=0, column=1, sticky="WE")
        self.y_vel.grid(row=0, column=2, sticky="WE")
        self.x_pos.grid(row=1, column=1, sticky="WE")
        self.y_pos.grid(row=1, column=2, sticky="WE")
        self.auto_rotate.grid(row=2, column=0, sticky="WES")
        self.angle.grid(row=2, column=1, columnspan=2, sticky="WES")
        self.rotate_speed.grid(row=2, column=2, columnspan=2, sticky="WE")
        self.scl_x.grid(row=3, column=1, sticky="WE")
        self.scl_y.grid(row=3, column=2, sticky="WE")
        self.vel_btn.grid(row=4, column=0, sticky="WE")
        self.pos_btn.grid(row=4, column=1, sticky="WE")
        self.sze_btn.grid(row=4, column=2, sticky="WE")
        self.ang_btn.grid(row=5, column=0, sticky="WE")
        self.trk_btn.grid(row=5, column=1, sticky="WE")
        self.del_btn.grid(row=5, column=2, sticky="WE")
        self.back_btn.grid(row=6, column=0, columnspan=6, sticky="NSEW")

        self.preview = Label(self, text="Video Feed", image=ImageTk.PhotoImage(image=Img.fromarray(np.ones((master.camera_config[1], master.camera_config[1])))))

        if self.image.follow_face:
            self.follow_face()
        else:
            self.preview.grid(row=7, column=0, columnspan=3, sticky="NEWS")

        threading.Thread(target=master.display, args=(self.preview,)).start()

    def change_velocity(self):
        self.images[self.master.current_image].force = [int(float(self.x_vel.get())), int(float(self.y_vel.get()))]

    def change_position(self):
        self.images[self.master.current_image].position = [int(self.x_pos.get()), int(self.y_pos.get())]

    def toggle_auto_rotate(self):
        self.image.auto_rotate = not self.image.auto_rotate

    def set_angle(self):
        self.image.angle = int(self.angle.get())

    def follow_face(self):
        self.preview.grid(row=9, column=0, columnspan=3, sticky="NEWS")
        self.follow_face_offset_label = Label(self, text="Follow Face Offset")
        self.follow_face_offset_x = Scale(self, from_=-128, to=128, orient=HORIZONTAL, command=lambda x=None: self.change_face_track_offset())
        self.follow_face_offset_y = Scale(self, from_=-128, to=128, orient=HORIZONTAL, command=lambda x=None: self.change_face_track_offset())

        self.follow_face_offset_label.grid(row=7, column=0, sticky="SWE")
        self.follow_face_offset_x.grid(row=7, column=1, sticky="SWE")
        self.follow_face_offset_y.grid(row=7, column=2, sticky="SWE")

    def set_rotate_speed(self):
        self.image.rotate_speed = self.var_rotate_speed.get()

    def change_face_track_offset(self):
        self.image.follow_face_offset = [self.follow_face_offset_x.get(), self.follow_face_offset_y.get()]

    def set_scale(self):
        self.image.position = [0, 0]
        self.image.base_image = cv2.resize(self.image.base_image, (int(self.scl_x.get()), int(self.scl_y.get())),
                                           interpolation=cv2.INTER_AREA)
        self.image.source = cv2.resize(self.image.base_image, (int(self.scl_x.get()), int(self.scl_y.get())),
                                       interpolation=cv2.INTER_AREA)

    def delete_image(self):
        del self.master.images[self.master.current_image]
        self.master.switch_frame(MainPage)


class Image:
    def __init__(self, src, name: str, position: list, force: list):
        self.index = None
        self.angle = 0
        self.position = position
        self.name = name
        self.follow_face = False
        self.follow_face_offset = [0, 0]

        if type(src) == np.ndarray:
            self.base_image = src
        else:
            self.base_image = cv2.imread(src, -1)

        self.source = cv2.resize(self.base_image, (64, 64))
        self.force = force

        self.auto_rotate = False
        self.rotate_speed = 1

        self.error = False

    def animate(self, canvas: np.array, master):
        if not self.follow_face:
            if self.position[0] + self.force[0] <= 0 or self.position[0] + self.force[0] + (self.source.shape[1]) >= \
                    canvas.shape[1]:
                self.force[0] = self.force[0] * -1
            if self.position[1] + self.force[1] <= 0 or self.position[1] + self.force[1] + (self.source.shape[0]) >= \
                    canvas.shape[0]:
                self.force[1] = self.force[1] * -1

            self.position = [self.position[0] + self.force[0], self.position[1] + self.force[1]]
        else:
            self.position[0] = master.face_info[1] - self.follow_face_offset[0]
            self.position[1] = master.face_info[2] - self.follow_face_offset[1]

        if self.auto_rotate:
            self.angle += self.rotate_speed

        self.rotate()
        self.display(canvas)

    def rotate(self):
        image_center = tuple(np.array(self.base_image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.0)
        self.source = cv2.warpAffine(self.base_image, rot_mat, self.base_image.shape[1::-1])

    def display(self, canvas: np.array):
        y1, y2 = self.position[1], self.position[1] + self.source.shape[0]
        x1, x2 = self.position[0], self.position[0] + self.source.shape[1]

        alpha_s = self.source[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            try:
                canvas[y1:y2, x1:x2, c] = (alpha_s * self.source[:, :, c] +
                                           alpha_l * canvas[y1:y2, x1:x2, c])
                self.error = False
            except Exception as error:
                print(error)
                self.error = True

    def get_position(self):
        return self.position

    def toggle_face_tracking(self):
        self.follow_face = not self.follow_face


# Optimize
def face_detection(feed: np.array, master):
    faces = face_cascade.detectMultiScale(feed, 1.1, 4)

    for (x, y, w, h) in faces:
        master.face_info = [len(faces), x, y, w, h]
        break
        # feed = cv2.rectangle(feed, (x, y), (x+w, y+h), (0, 255, 0), 2)


def main():
    cv2.setNumThreads(4)
    application = Application()
    application.title("Virtual Camera Changer")
    application.resizable(False, False)
    application.mainloop()
    sys.exit()


if __name__ == '__main__':
    main()
