"""Make PyOpenGL draw through the same OpenGL library that Qt picked.

On Windows machines without a proper graphics driver (VMs, remote desktop, old
PCs) Qt quietly switches to its bundled software renderer, opengl32sw.dll, but
PyOpenGL still calls the system opengl32.dll, which then has no current
context: every call fails and the 3D preview stays black. Pointing PyOpenGL at
Qt's library fixes that. Must run after the QApplication exists and before
anything imports OpenGL.GL (pyqtgraph.opengl does).
"""

import ctypes
import sys


def match_qt_opengl():
    if sys.platform != "win32":
        return
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext

    # Creating a context is what makes Qt choose (and load) its OpenGL library.
    surface = QOffscreenSurface()
    surface.create()
    context = QOpenGLContext()
    if not context.create() or not context.makeCurrent(surface):
        return
    context.doneCurrent()

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    handle = kernel32.GetModuleHandleW("opengl32sw.dll")
    if not handle:
        return  # Qt is using the real driver, same as PyOpenGL.

    from OpenGL import platform

    # Importing OpenGL.platform already resolved helpers such as GetCurrentContext from
    # the system DLL and copied them into the module; drop those cached values, swap
    # the library, and install again so everything is resolved from Qt's library.
    plat = platform.PLATFORM
    vars(plat).clear()
    plat.GL = plat.OpenGL = ctypes.WinDLL("opengl32sw.dll", handle=handle)
    plat.install(vars(platform))
