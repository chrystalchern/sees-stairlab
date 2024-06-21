# Claudio Perez
# Summer 2024
import sys
import bottle
from .viewer import Viewer

class Server:
    def __init__(self, glb=None, html=None, viewer=None):
        if html is None:
            self._page = Viewer(src="./model.glb",
                                viewer=viewer).get_html()
        else:
            self._page = html

        # Create App
        self._app = bottle.Bottle()
        self._app.route("/")(lambda : self._page )

        if glb is not None:
            self._app.route("/model.glb")(lambda : glb)

    def run(self, port=None):
        if port is None:
            port = 8080

        try:
            bottle.run(self._app, host="localhost", port=port)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":

    options = {
        "viewer": None
    }
    filename = sys.argv[1]

    with open(filename, "rb") as f:
        glb = f.read()

    Server(glb, **options).run()


