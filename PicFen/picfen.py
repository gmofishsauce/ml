import cv2
import numpy as np
import chess
import sys

# ------------------------------
# Configuration
# ------------------------------
# Folder of templates (white_pawn.png, black_queen.png, etc.)
TEMPLATE_DIR = "templates/"

# File name of scan
SCAN_FILE = "page.jpg"

# If true, print diagrams and templates, exit, don't do any analysis
print_diagrams_and_templates = False

# ------------------------------
# 1. Load and preprocess page
# ------------------------------
page = cv2.imread(SCAN_FILE, cv2.IMREAD_GRAYSCALE)
page = cv2.GaussianBlur(page, (3,3), 0)
_, page_bin = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ------------------------------
# 2. Detect diagram boxes (3x2 grid)
# ------------------------------
# Find contours
contours, _ = cv2.findContours(page_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort by area, assume six largest rectangular contours are diagrams
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

# Extract bounding boxes
boxes = [cv2.boundingRect(c) for c in contours]

# Sort boxes top-to-bottom, left-to-right
boxes = sorted(boxes, key=lambda b: (b[1]//100, b[0]))
print(f"boxes: {boxes}")

# The board images are surrounded by a border. The
# size was estimated by trial and error.
BORDER_WIDTH = 8

# --------------------------------------
# 3. Optional: print diagrams and templatess, exit
# --------------------------------------

if print_diagrams_and_templates:
    # Print all six diagrams. The diagrams start with #307
    # in the book, so we call the files 307.png, etc.
    diagram_base = 307

    for diagram_number in range(6):
        x, y, w, h = boxes[diagram_number]
        diagram = page[y+BORDER_WIDTH:y+h-BORDER_WIDTH, x+BORDER_WIDTH:x+w-BORDER_WIDTH]
        # cv2.imwrite(f"{diagram_base+diagram_number}.png", diagram)
        # print(f"wrote {diagram_base+diagram_number}.png")

        # Detect internal 8x8 grid
        # Assume clean rectangular board
        H, W = diagram.shape
        square_h = H // 8
        square_w = W // 8

        # write template files
        for row in range(8):
            for col in range(8):
                y0, y1 = row*square_h, (row+1)*square_h
                x0, x1 = col*square_w, (col+1)*square_w
                cell = diagram[y0:y1, x0:x1]
                cv2.imwrite(f"templates/{diagram_number}-{row}-{col}.png", cell)
        print("wrote 64 template files")
        diagram_number += 1
    # templates written - done.
    sys.exit(5)

# ------------------------------
# 4. Load templates
# ------------------------------
PIECES = ["WP","WN","WB","WR","WQ","WK","Bp","Bn","Bb","Br","Bq","Bk"]
BACKGROUNDS = ["L", "D"]
templates = {}
tmpl_index = 0
for p in PIECES:
    for s in BACKGROUNDS:
        tmpl = cv2.imread(f"{TEMPLATE_DIR}/{p}{s}.png", cv2.IMREAD_GRAYSCALE)
        #templates[f"{p}{s}"] = cv2.resize(tmpl, (square_w, square_h))
        templates[f"{p}{s}"] = tmpl
        tmpl_index += 1

tmpl = cv2.imread(f"{TEMPLATE_DIR}/XxD.png", cv2.IMREAD_GRAYSCALE)
templates["XxD"] = tmpl
tmpl_index += 1

tmpl = cv2.imread(f"{TEMPLATE_DIR}/XxL.png", cv2.IMREAD_GRAYSCALE)
templates["XxL"] = tmpl
tmpl_index += 1

# ------------------------------
# 5. Classify each square
# ------------------------------
for diagram_number in range(6):
    x, y, w, h = boxes[diagram_number]
    diagram = page[y+BORDER_WIDTH:y+h-BORDER_WIDTH, x+BORDER_WIDTH:x+w-BORDER_WIDTH]
    H, W = diagram.shape
    square_h = H // 8
    square_w = W // 8

    board = [["" for _ in range(8)] for _ in range(8)]
    scores = [[0.0 for _ in range(8)] for _ in range(8)]
    
    for row in range(8):
        for col in range(8):
            y0, y1 = row*square_h, (row+1)*square_h
            x0, x1 = col*square_w, (col+1)*square_w
            cell = diagram[y0:y1, x0:x1]

            best_piece = ""
            # best_score = 0.0
            best_score = 1.0

            for piece, tmpl in templates.items():
                #res = cv2.matchTemplate(cell, tmpl, cv2.TM_CCOEFF_NORMED)
                res = cv2.matchTemplate(cell, tmpl, cv2.TM_SQDIFF_NORMED)
                # _, score, _, _ = cv2.minMaxLoc(res)
                score, _, _, _ = cv2.minMaxLoc(res)

                # print(f"{piece} = {score:.2f}")

                if score < best_score:
                    best_piece, best_score = piece, score

            # print('')
            board[row][col] = best_piece
            scores[row][col] = best_score

    for row in range(8):
        for col in range(8):
            print(f"{board[row][col]}({scores[row][col]:.2f})", end=' ')
        print("")
    print("")

sys.exit(5)

# ------------------------------
# 6. Convert to FEN
# ------------------------------
# Note: images usually have rank 8 at top → must reverse rows
rows = []
for r in range(8):
    fen_row = ""
    empty = 0
    for c in range(8):
        piece = board[r][c]
        if piece == "":
            empty += 1
        else:
            if empty:
                fen_row += str(empty)
                empty = 0
            fen_row += piece
    if empty:
        fen_row += str(empty)
    rows.append(fen_row)

# Reverse order so rank 8 (top) → first in FEN
fen = "/".join(rows) + " w - - 0 1"
print(f"FEN: {fen}")

# ------------------------------
# 7. Validate
# ------------------------------
import chess.svg
board_obj = chess.Board(fen)
print(board_obj.unicode())

