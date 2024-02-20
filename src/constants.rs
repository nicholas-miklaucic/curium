//! Module for constants.

/// GROUP_ORDERS[i] is the order of space group i.
pub const GROUP_ORDERS: [usize; 231] = [
    0, 1, 2, 2, 2, 4, 2, 2, 4, 4, 4, 4, 8, 4, 4, 8, 4, 4, 4, 4, 8, 8, 16, 8, 8, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 16, 16, 16, 16, 16, 16, 32, 32, 16, 16, 16, 16, 4, 4, 4, 4, 8, 8, 4, 8, 8, 8, 8, 8, 16, 16,
    8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32,
    3, 3, 3, 9, 6, 18, 6, 6, 6, 6, 6, 6, 18, 6, 6, 6, 6, 18, 18, 12, 12, 12, 12, 36, 36, 6, 6, 6,
    6, 6, 6, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 12,
    48, 24, 12, 24, 24, 24, 96, 96, 48, 24, 48, 24, 24, 96, 96, 48, 24, 24, 48, 24, 96, 48, 24, 96,
    48, 48, 48, 48, 48, 192, 192, 192, 192, 96, 96,
];

/// A tuple matching hall symbols and their space group.
pub const HALL_SYMBOLS: [(&str, usize); 523] = [
    ("P 1", 1),
    ("-P 1", 2),
    ("P 2y", 3),
    ("P 2", 3),
    ("P 2x", 3),
    ("P 2yb", 4),
    ("P 2c", 4),
    ("P 2xa", 4),
    ("C 2y", 5),
    ("A 2y", 5),
    ("I 2y", 5),
    ("A 2", 5),
    ("B 2", 5),
    ("I 2", 5),
    ("B 2x", 5),
    ("C 2x", 5),
    ("I 2x", 5),
    ("P -2y", 6),
    ("P -2", 6),
    ("P -2x", 6),
    ("P -2yc", 7),
    ("P -2yac", 7),
    ("P -2ya", 7),
    ("P -2a", 7),
    ("P -2ab", 7),
    ("P -2b", 7),
    ("P -2xb", 7),
    ("P -2xbc", 7),
    ("P -2xc", 7),
    ("C -2y", 8),
    ("A -2y", 8),
    ("I -2y", 8),
    ("A -2", 8),
    ("B -2", 8),
    ("I -2", 8),
    ("B -2x", 8),
    ("C -2x", 8),
    ("I -2x", 8),
    ("C -2yc", 9),
    ("A -2yac", 9),
    ("I -2ya", 9),
    ("A -2ya", 9),
    ("C -2ybc", 9),
    ("I -2yc", 9),
    ("A -2a", 9),
    ("B -2bc", 9),
    ("I -2b", 9),
    ("B -2b", 9),
    ("A -2ac", 9),
    ("I -2a", 9),
    ("B -2xb", 9),
    ("C -2xbc", 9),
    ("I -2xc", 9),
    ("C -2xc", 9),
    ("B -2xbc", 9),
    ("I -2xb", 9),
    ("-P 2y", 10),
    ("-P 2", 10),
    ("-P 2x", 10),
    ("-P 2yb", 11),
    ("-P 2c", 11),
    ("-P 2xa", 11),
    ("-C 2y", 12),
    ("-A 2y", 12),
    ("-I 2y", 12),
    ("-A 2", 12),
    ("-B 2", 12),
    ("-I 2", 12),
    ("-B 2x", 12),
    ("-C 2x", 12),
    ("-I 2x", 12),
    ("-P 2yc", 13),
    ("-P 2yac", 13),
    ("-P 2ya", 13),
    ("-P 2a", 13),
    ("-P 2ab", 13),
    ("-P 2b", 13),
    ("-P 2xb", 13),
    ("-P 2xbc", 13),
    ("-P 2xc", 13),
    ("-P 2ybc", 14),
    ("-P 2yn", 14),
    ("-P 2yab", 14),
    ("-P 2ac", 14),
    ("-P 2n", 14),
    ("-P 2bc", 14),
    ("-P 2xab", 14),
    ("-P 2xn", 14),
    ("-P 2xac", 14),
    ("-C 2yc", 15),
    ("-A 2yac", 15),
    ("-I 2ya", 15),
    ("-A 2ya", 15),
    ("-C 2ybc", 15),
    ("-I 2yc", 15),
    ("-A 2a", 15),
    ("-B 2bc", 15),
    ("-I 2b", 15),
    ("-B 2b", 15),
    ("-A 2ac", 15),
    ("-I 2a", 15),
    ("-B 2xb", 15),
    ("-C 2xbc", 15),
    ("-I 2xc", 15),
    ("-C 2xc", 15),
    ("-B 2xbc", 15),
    ("-I 2xb", 15),
    ("P 2 2", 16),
    ("P 2c 2", 17),
    ("P 2a 2a", 17),
    ("P 2 2b", 17),
    ("P 2 2ab", 18),
    ("P 2bc 2", 18),
    ("P 2ac 2ac", 18),
    ("P 2ac 2ab", 19),
    ("C 2c 2", 20),
    ("A 2a 2a", 20),
    ("B 2 2b", 20),
    ("C 2 2", 21),
    ("A 2 2", 21),
    ("B 2 2", 21),
    ("F 2 2", 22),
    ("I 2 2", 23),
    ("I 2b 2c", 24),
    ("P 2 -2", 25),
    ("P -2 2", 25),
    ("P -2 -2", 25),
    ("P 2c -2", 26),
    ("P 2c -2c", 26),
    ("P -2a 2a", 26),
    ("P -2 2a", 26),
    ("P -2 -2b", 26),
    ("P -2b -2", 26),
    ("P 2 -2c", 27),
    ("P -2a 2", 27),
    ("P -2b -2b", 27),
    ("P 2 -2a", 28),
    ("P 2 -2b", 28),
    ("P -2b 2", 28),
    ("P -2c 2", 28),
    ("P -2c -2c", 28),
    ("P -2a -2a", 28),
    ("P 2c -2ac", 29),
    ("P 2c -2b", 29),
    ("P -2b 2a", 29),
    ("P -2ac 2a", 29),
    ("P -2bc -2c", 29),
    ("P -2a -2ab", 29),
    ("P 2 -2bc", 30),
    ("P 2 -2ac", 30),
    ("P -2ac 2", 30),
    ("P -2ab 2", 30),
    ("P -2ab -2ab", 30),
    ("P -2bc -2bc", 30),
    ("P 2ac -2", 31),
    ("P 2bc -2bc", 31),
    ("P -2ab 2ab", 31),
    ("P -2 2ac", 31),
    ("P -2 -2bc", 31),
    ("P -2ab -2", 31),
    ("P 2 -2ab", 32),
    ("P -2bc 2", 32),
    ("P -2ac -2ac", 32),
    ("P 2c -2n", 33),
    ("P 2c -2ab", 33),
    ("P -2bc 2a", 33),
    ("P -2n 2a", 33),
    ("P -2n -2ac", 33),
    ("P -2ac -2n", 33),
    ("P 2 -2n", 34),
    ("P -2n 2", 34),
    ("P -2n -2n", 34),
    ("C 2 -2", 35),
    ("A -2 2", 35),
    ("B -2 -2", 35),
    ("C 2c -2", 36),
    ("C 2c -2c", 36),
    ("A -2a 2a", 36),
    ("A -2 2a", 36),
    ("B -2 -2b", 36),
    ("B -2b -2", 36),
    ("C 2 -2c", 37),
    ("A -2a 2", 37),
    ("B -2b -2b", 37),
    ("A 2 -2", 38),
    ("B 2 -2", 38),
    ("B -2 2", 38),
    ("C -2 2", 38),
    ("C -2 -2", 38),
    ("A -2 -2", 38),
    ("A 2 -2c", 39),
    ("B 2 -2c", 39),
    ("B -2c 2", 39),
    ("C -2b 2", 39),
    ("C -2b -2b", 39),
    ("A -2c -2c", 39),
    ("A 2 -2a", 40),
    ("B 2 -2b", 40),
    ("B -2b 2", 40),
    ("C -2c 2", 40),
    ("C -2c -2c", 40),
    ("A -2a -2a", 40),
    ("A 2 -2ac", 41),
    ("B 2 -2bc", 41),
    ("B -2bc 2", 41),
    ("C -2bc 2", 41),
    ("C -2bc -2bc", 41),
    ("A -2ac -2ac", 41),
    ("F 2 -2", 42),
    ("F -2 2", 42),
    ("F -2 -2", 42),
    ("F 2 -2d", 43),
    ("F -2d 2", 43),
    ("F -2d -2d", 43),
    ("I 2 -2", 44),
    ("I -2 2", 44),
    ("I -2 -2", 44),
    ("I 2 -2c", 45),
    ("I -2a 2", 45),
    ("I -2b -2b", 45),
    ("I 2 -2a", 46),
    ("I 2 -2b", 46),
    ("I -2b 2", 46),
    ("I -2c 2", 46),
    ("I -2c -2c", 46),
    ("I -2a -2a", 46),
    ("-P 2 2", 47),
    ("P 2 2 -1n", 48),
    ("-P 2ab 2bc", 48),
    ("-P 2 2c", 49),
    ("-P 2a 2", 49),
    ("-P 2b 2b", 49),
    ("P 2 2 -1ab", 50),
    ("-P 2ab 2b", 50),
    ("P 2 2 -1bc", 50),
    ("-P 2b 2bc", 50),
    ("P 2 2 -1ac", 50),
    ("-P 2a 2c", 50),
    ("-P 2a 2a", 51),
    ("-P 2b 2", 51),
    ("-P 2 2b", 51),
    ("-P 2c 2c", 51),
    ("-P 2c 2", 51),
    ("-P 2 2a", 51),
    ("-P 2a 2bc", 52),
    ("-P 2b 2n", 52),
    ("-P 2n 2b", 52),
    ("-P 2ab 2c", 52),
    ("-P 2ab 2n", 52),
    ("-P 2n 2bc", 52),
    ("-P 2ac 2", 53),
    ("-P 2bc 2bc", 53),
    ("-P 2ab 2ab", 53),
    ("-P 2 2ac", 53),
    ("-P 2 2bc", 53),
    ("-P 2ab 2", 53),
    ("-P 2a 2ac", 54),
    ("-P 2b 2c", 54),
    ("-P 2a 2b", 54),
    ("-P 2ac 2c", 54),
    ("-P 2bc 2b", 54),
    ("-P 2b 2ab", 54),
    ("-P 2 2ab", 55),
    ("-P 2bc 2", 55),
    ("-P 2ac 2ac", 55),
    ("-P 2ab 2ac", 56),
    ("-P 2ac 2bc", 56),
    ("-P 2bc 2ab", 56),
    ("-P 2c 2b", 57),
    ("-P 2c 2ac", 57),
    ("-P 2ac 2a", 57),
    ("-P 2b 2a", 57),
    ("-P 2a 2ab", 57),
    ("-P 2bc 2c", 57),
    ("-P 2 2n", 58),
    ("-P 2n 2", 58),
    ("-P 2n 2n", 58),
    ("P 2 2ab -1ab", 59),
    ("-P 2ab 2a", 59),
    ("P 2bc 2 -1bc", 59),
    ("-P 2c 2bc", 59),
    ("P 2ac 2ac -1ac", 59),
    ("-P 2c 2a", 59),
    ("-P 2n 2ab", 60),
    ("-P 2n 2c", 60),
    ("-P 2a 2n", 60),
    ("-P 2bc 2n", 60),
    ("-P 2ac 2b", 60),
    ("-P 2b 2ac", 60),
    ("-P 2ac 2ab", 61),
    ("-P 2bc 2ac", 61),
    ("-P 2ac 2n", 62),
    ("-P 2bc 2a", 62),
    ("-P 2c 2ab", 62),
    ("-P 2n 2ac", 62),
    ("-P 2n 2a", 62),
    ("-P 2c 2n", 62),
    ("-C 2c 2", 63),
    ("-C 2c 2c", 63),
    ("-A 2a 2a", 63),
    ("-A 2 2a", 63),
    ("-B 2 2b", 63),
    ("-B 2b 2", 63),
    ("-C 2bc 2", 64),
    ("-C 2bc 2bc", 64),
    ("-A 2ac 2ac", 64),
    ("-A 2 2ac", 64),
    ("-B 2 2bc", 64),
    ("-B 2bc 2", 64),
    ("-C 2 2", 65),
    ("-A 2 2", 65),
    ("-B 2 2", 65),
    ("-C 2 2c", 66),
    ("-A 2a 2", 66),
    ("-B 2b 2b", 66),
    ("-C 2b 2", 67),
    ("-C 2b 2b", 67),
    ("-A 2c 2c", 67),
    ("-A 2 2c", 67),
    ("-B 2 2c", 67),
    ("-B 2c 2", 67),
    ("C 2 2 -1bc", 68),
    ("-C 2b 2bc", 68),
    ("C 2 2 -1bc", 68),
    ("-C 2b 2c", 68),
    ("A 2 2 -1ac", 68),
    ("-A 2a 2c", 68),
    ("A 2 2 -1ac", 68),
    ("-A 2ac 2c", 68),
    ("B 2 2 -1bc", 68),
    ("-B 2bc 2b", 68),
    ("B 2 2 -1bc", 68),
    ("-B 2b 2bc", 68),
    ("-F 2 2", 69),
    ("F 2 2 -1d", 70),
    ("-F 2uv 2vw", 70),
    ("-I 2 2", 71),
    ("-I 2 2c", 72),
    ("-I 2a 2", 72),
    ("-I 2b 2b", 72),
    ("-I 2b 2c", 73),
    ("-I 2a 2b", 73),
    ("-I 2b 2", 74),
    ("-I 2a 2a", 74),
    ("-I 2c 2c", 74),
    ("-I 2 2b", 74),
    ("-I 2 2a", 74),
    ("-I 2c 2", 74),
    ("P 4", 75),
    ("P 4w", 76),
    ("P 4c", 77),
    ("P 4cw", 78),
    ("I 4", 79),
    ("I 4bw", 80),
    ("P -4", 81),
    ("I -4", 82),
    ("-P 4", 83),
    ("-P 4c", 84),
    ("P 4ab -1ab", 85),
    ("-P 4a", 85),
    ("P 4n -1n", 86),
    ("-P 4bc", 86),
    ("-I 4", 87),
    ("I 4bw -1bw", 88),
    ("-I 4ad", 88),
    ("P 4 2", 89),
    ("P 4ab 2ab", 90),
    ("P 4w 2c", 91),
    ("P 4abw 2nw", 92),
    ("P 4c 2", 93),
    ("P 4n 2n", 94),
    ("P 4cw 2c", 95),
    ("P 4nw 2abw", 96),
    ("I 4 2", 97),
    ("I 4bw 2bw", 98),
    ("P 4 -2", 99),
    ("P 4 -2ab", 100),
    ("P 4c -2c", 101),
    ("P 4n -2n", 102),
    ("P 4 -2c", 103),
    ("P 4 -2n", 104),
    ("P 4c -2", 105),
    ("P 4c -2ab", 106),
    ("I 4 -2", 107),
    ("I 4 -2c", 108),
    ("I 4bw -2", 109),
    ("I 4bw -2c", 110),
    ("P -4 2", 111),
    ("P -4 2c", 112),
    ("P -4 2ab", 113),
    ("P -4 2n", 114),
    ("P -4 -2", 115),
    ("P -4 -2c", 116),
    ("P -4 -2ab", 117),
    ("P -4 -2n", 118),
    ("I -4 -2", 119),
    ("I -4 -2c", 120),
    ("I -4 2", 121),
    ("I -4 2bw", 122),
    ("-P 4 2", 123),
    ("-P 4 2c", 124),
    ("P 4 2 -1ab", 125),
    ("-P 4a 2b", 125),
    ("P 4 2 -1n", 126),
    ("-P 4a 2bc", 126),
    ("-P 4 2ab", 127),
    ("-P 4 2n", 128),
    ("P 4ab 2ab -1ab", 129),
    ("-P 4a 2a", 129),
    ("P 4ab 2n -1ab", 130),
    ("-P 4a 2ac", 130),
    ("-P 4c 2", 131),
    ("-P 4c 2c", 132),
    ("P 4n 2c -1n", 133),
    ("-P 4ac 2b", 133),
    ("P 4n 2 -1n", 134),
    ("-P 4ac 2bc", 134),
    ("-P 4c 2ab", 135),
    ("-P 4n 2n", 136),
    ("P 4n 2n -1n", 137),
    ("-P 4ac 2a", 137),
    ("P 4n 2ab -1n", 138),
    ("-P 4ac 2ac", 138),
    ("-I 4 2", 139),
    ("-I 4 2c", 140),
    ("I 4bw 2bw -1bw", 141),
    ("-I 4bd 2", 141),
    ("I 4bw 2aw -1bw", 142),
    ("-I 4bd 2c", 142),
    ("P 3", 143),
    ("P 31", 144),
    ("P 32", 145),
    ("R 3", 146),
    // ("P 3*", 146),
    ("-P 3", 147),
    ("-R 3", 148),
    // ("-P 3*", 148),
    ("P 3 2", 149),
    ("P 3 2\"", 150),
    ("P 31 2c (0 0 1)", 151),
    ("P 31 2\"", 152),
    ("P 32 2c (0 0 -1)", 153),
    ("P 32 2\"", 154),
    ("R 3 2\"", 155),
    // ("P 3* 2", 155),
    ("P 3 -2\"", 156),
    ("P 3 -2", 157),
    ("P 3 -2\"c", 158),
    ("P 3 -2c", 159),
    ("R 3 -2\"", 160),
    // ("P 3* -2", 160),
    ("R 3 -2\"c", 161),
    // ("P 3* -2n", 161),
    ("-P 3 2", 162),
    ("-P 3 2c", 163),
    ("-P 3 2\"", 164),
    ("-P 3 2\"c", 165),
    ("-R 3 2\"", 166),
    // ("-P 3* 2", 166),
    ("-R 3 2\"c", 167),
    // ("-P 3* 2n", 167),
    ("P 6", 168),
    ("P 61", 169),
    ("P 65", 170),
    ("P 62", 171),
    ("P 64", 172),
    ("P 6c", 173),
    ("P -6", 174),
    ("-P 6", 175),
    ("-P 6c", 176),
    ("P 6 2", 177),
    ("P 61 2 (0 0 -1)", 178),
    ("P 65 2 (0 0 1)", 179),
    ("P 62 2c (0 0 1)", 180),
    ("P 64 2c (0 0 -1)", 181),
    ("P 6c 2c", 182),
    ("P 6 -2", 183),
    ("P 6 -2c", 184),
    ("P 6c -2", 185),
    ("P 6c -2c", 186),
    ("P -6 2", 187),
    ("P -6c 2", 188),
    ("P -6 -2", 189),
    ("P -6c -2c", 190),
    ("-P 6 2", 191),
    ("-P 6 2c", 192),
    ("-P 6c 2", 193),
    ("-P 6c 2c", 194),
    ("P 2 2 3", 195),
    ("F 2 2 3", 196),
    ("I 2 2 3", 197),
    ("P 2ac 2ab 3", 198),
    ("I 2b 2c 3", 199),
    ("-P 2 2 3", 200),
    ("P 2 2 3 -1n", 201),
    ("-P 2ab 2bc 3", 201),
    ("-F 2 2 3", 202),
    ("F 2 2 3 -1d", 203),
    ("-F 2uv 2vw 3", 203),
    ("-I 2 2 3", 204),
    ("-P 2ac 2ab 3", 205),
    ("-I 2b 2c 3", 206),
    ("P 4 2 3", 207),
    ("P 4n 2 3", 208),
    ("F 4 2 3", 209),
    ("F 4d 2 3", 210),
    ("I 4 2 3", 211),
    ("P 4acd 2ab 3", 212),
    ("P 4bd 2ab 3", 213),
    ("I 4bd 2c 3", 214),
    ("P -4 2 3", 215),
    ("F -4 2 3", 216),
    ("I -4 2 3", 217),
    ("P -4n 2 3", 218),
    ("F -4c 2 3", 219),
    ("I -4bd 2c 3", 220),
    ("-P 4 2 3", 221),
    ("P 4 2 3 -1n", 222),
    ("-P 4a 2bc 3", 222),
    ("-P 4n 2 3", 223),
    ("P 4n 2 3 -1n", 224),
    ("-P 4bc 2bc 3", 224),
    ("-F 4 2 3", 225),
    ("-F 4c 2 3", 226),
    ("F 4d 2 3 -1d", 227),
    ("-F 4vw 2vw 3", 227),
    ("F 4d 2 3 -1cd", 228),
    ("-F 4cvw 2vw 3", 228),
    ("-I 4 2 3", 229),
    ("-I 4bd 2c 3", 230),
];

/// The short Hermann-Mauguin symbols in order, represented using ASCII, no underscores for screws,
/// and with hexagonal axes for the rhombohedral groups.
pub const SPACE_GROUP_SYMBOLS: [&str; 230] = [
    "P1", "P-1", "P2", "P21", "C2", "Pm", "Pc", "Cm", "Cc", "P2/m", "P21/m", "C2/m", "P2/c",
    "P21/c", "C2/c", "P222", "P2221", "P21212", "P212121", "C2221", "C222", "F222", "I222",
    "I212121", "Pmm2", "Pmc21", "Pcc2", "Pma2", "Pca21", "Pnc2", "Pmn21", "Pba2", "Pna21", "Pnn2",
    "Cmm2", "Cmc21", "Ccc2", "Amm2", "Aem2", "Ama2", "Aea2", "Fmm2", "Fdd2", "Imm2", "Iba2",
    "Ima2", "Pmmm", "Pnnn", "Pccm", "Pban", "Pmma", "Pnna", "Pmna", "Pcca", "Pbam", "Pccn", "Pbcm",
    "Pnnm", "Pmmn", "Pbcn", "Pbca", "Pnma", "Cmcm", "Cmce", "Cmmm", "Cccm", "Cmme", "Ccce", "Fmmm",
    "Fddd", "Immm", "Ibam", "Ibca", "Imma", "P4", "P41", "P42", "P43", "I4", "I41", "P-4", "I-4",
    "P4/m", "P42/m", "P4/n", "P42/n", "I4/m", "I41/a", "P422", "P4212", "P4122", "P41212", "P4222",
    "P42212", "P4322", "P43212", "I422", "I4122", "P4mm", "P4bm", "P42cm", "P42nm", "P4cc", "P4nc",
    "P42mc", "P42bc", "I4mm", "I4cm", "I41md", "I41cd", "P-42m", "P-42c", "P-421m", "P-421c",
    "P-4m2", "P-4c2", "P-4b2", "P-4n2", "I-4m2", "I-4c2", "I-42m", "I-42d", "P4/mmm", "P4/mcc",
    "P4/nbm", "P4/nnc", "P4/mbm", "P4/mnc", "P4/nmm", "P4/ncc", "P42/mmc", "P42/mcm", "P42/nbc",
    "P42/nnm", "P42/mbc", "P42/mnm", "P42/nmc", "P42/ncm", "I4/mmm", "I4/mcm", "I41/amd",
    "I41/acd", "P3", "P31", "P32", "R3", "P-3", "R-3", "P312", "P321", "P3112", "P3121", "P3212",
    "P3221", "R32", "P3m1", "P31m", "P3c1", "P31c", "R3m", "R3c", "P-31m", "P-31c", "P-3m1",
    "P-3c1", "R-3m", "R-3c", "P6", "P61", "P65", "P62", "P64", "P63", "P-6", "P6/m", "P63/m",
    "P622", "P6122", "P6522", "P6222", "P6422", "P6322", "P6mm", "P6cc", "P63cm", "P63mc", "P-6m2",
    "P-6c2", "P-62m", "P-62c", "P6/mmm", "P6/mcc", "P63/mcm", "P63/mmc", "P23", "F23", "I23",
    "P213", "I213", "Pm-3", "Pn-3", "Fm-3", "Fd-3", "Im-3", "Pa-3", "Ia-3", "P432", "P4232",
    "F432", "F4132", "I432", "P4332", "P4132", "I4132", "P-43m", "F-43m", "I-43m", "P-43n",
    "F-43c", "I-43d", "Pm-3m", "Pn-3n", "Pm-3n", "Pn-3m", "Fm-3m", "Fm-3c", "Fd-3m", "Fd-3c",
    "Im-3m", "Ia-3d",
];