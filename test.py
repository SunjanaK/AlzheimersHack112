
from tkinter import *



#strategies: take the big problem and divide it into smaller chunks based on the various aspects of the problem(ex: color assignment, location, size), don't be afraid to use extra arguments in your helper functions
def drawCircles(n, canvas, width, height):
    r = 20
    canvas.create_rectangle(0, 0, width, height, fill="blue")
    d = 2*r
    for i in range(n):
        for c in range(n):
            xleft = c * d
            yleft= i * d
            xright = c * d + d
            yright = i * d + d
            drawBullseye(xleft, yleft, xright, yright, canvas, n, r,i,c)

def drawBullseye(xl, yl, xr, yr, canvas, n, r,row,col):
    color = colorAssign(n, xl, yl, xr, yr, row, col)
    canvas.create_oval(xl, yl, xr, yr, fill = color)
    decrease = 2/3

    while r > 2:
        r = decrease * r
        canvas.create_oval(xl, yl, xr, yr, fill = color)

def colorAssign(n, xl, yl, xr, yr, row, col):
    if (row + col) % 4 == 0:
        color = "red"
        return color
    elif row % 3 == 0:
        color = "green"
        return color
    elif col % 2 == 1:
        color = "yellow"
        return color
    else:
        color = "blue"
        return color
def drawCirclePattern(n, winWidth=500,winHeight=500):
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    canvas = Canvas(root, width=winWidth, height=winHeight)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    drawCircles(n, canvas, winWidth, winHeight)
    root.mainloop()

drawCirclePattern(5)
