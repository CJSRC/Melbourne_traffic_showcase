"""The four prompts, one per job, each written in blocks so a section can be tuned alone.

Every one describes what is actually sent: a single satellite photograph with the massing
solid already painted onto it, plus an alpha mask marking the region the model may repaint.
The prompt that had been in use described two images and a grey massing study -- that is what
the older two-image pipeline sent, and the webapp has been sending one composited image since,
so the model was being told about a picture it never received.

Blocks are separated by a blank line and headed in capitals. The model does not need the
headings; we do. A shadow problem gets fixed in LIGHT without touching GEOMETRY, and a diff
between two versions says which job changed.

Placeholders: {roof_word} {wall_word} follow the massing style, {height} {storeys} {area}
{use} describe the new building, {existing} the one being replaced or removed, and {count}
{buildings} the multi-block case.
"""

ADD_ONE = """SCENE
This is an overhead satellite photograph of a site in Melbourne. A massing study of an
approved building has been painted onto it at the exact position, footprint and scale the
building will occupy. The {roof_word} face is its roof; the {wall_word} faces are its sides.
Those two colours identify the surfaces and are not the building's materials.

TASK
Replace the painted solid with a photographic building, so the result reads as an ordinary
satellite photograph in which that building exists.

GEOMETRY
The ground position, the footprint outline and the roof outline are fixed. Do not move,
rotate, resize or simplify them. The building is {height:.0f} m tall, about {storeys} storeys,
on a footprint of {area:.0f} square metres, land use {use}.

The one {roof_word} face is the roof and the only roof. Every {wall_word} face is
a vertical wall seen from the side, not roof: put no plant, no services and no roof texture on
it. Telling them apart is a matter of position, not size -- at this height the walls can cover
more of the picture than the roof does, and they are still walls. The roof is the {roof_word}
outline, offset about {shift:.0f} m towards the {bearing} from the ground outline, because a
satellite {lean:.0f} degrees off vertical sees the top of a {height:.0f} m building displaced
that far. The ground outline is where the building meets the ground.

Show facade on the same side and to the same extent the solid does, and leave the roof and the
ground footprint exactly where they are drawn.

LIGHT
Every shadow in this frame runs the same direction, and the length of each relative to its own
building tells you the sun elevation. Read both off the real buildings in frame rather than
assuming them, then give the new building a shadow of matching direction and proportionate
length. Where that shadow falls across a road, a roof or a footpath, darken what is there
rather than painting a flat shape. Where the new building blocks light that used to reach the
ground, remove the shadow that no longer belongs.

SURFACE
Give it the substance a real building has at this resolution: roof plant, services, lift
overruns and parapets on the roof; glazing lines, spandrels and material on the facades. Match
the grain, sharpness, colour balance and haze of the surrounding photograph. The edges where it
meets the ground, the neighbouring roofs and the street should look photographed, not cut out.

Neither {roof_word} nor {wall_word} may appear in the result. They are markings, not materials: the finished roof is the grey of membrane, metal deck or concrete with its plant and services, and the finished walls are glazing, precast or brick. A {roof_word} roof or a {wall_word} wall in the output means the marking was copied instead of read.

HOLD
Apart from the new building and the shadows it changes, every pixel stays as it is. Add no
outline, label, annotation or watermark anywhere."""


ADD_MANY = """SCENE
This is an overhead satellite photograph of a site in Melbourne. Massing studies of {count}
approved buildings have been painted onto it at the exact positions, footprints and scales they
will occupy. The {roof_word} faces are roofs; the {wall_word} faces are sides. Those two
colours identify the surfaces and are not the buildings' materials.

TASK
Replace the painted solids with photographic buildings, so the result reads as an ordinary
satellite photograph in which all {count} exist.

GEOMETRY
Ground positions, footprint outlines and roof outlines are fixed. Do not move, rotate, resize,
merge or simplify them. They stay separate buildings and the gaps between them are preserved.
Where two solids overlap, the one painted in front stands in front. The buildings are:

{buildings}

Each {roof_word} face is a roof and the only roof of its building. Every {wall_word} face is a
vertical wall seen from the side, not roof: put no plant, no services and no roof texture on
it. At these heights the walls can cover more of the picture than the roofs do, and they are
still walls. Each roof outline is offset towards the {bearing} from its own ground outline,
further for a taller building, because the satellite is {lean:.0f} degrees off vertical. Show
facade on the same side and to the same extent each solid does, and leave every roof and ground
footprint exactly where it is drawn.

LIGHT
Every shadow in this frame runs the same direction, and the length of each relative to its own
building tells you the sun elevation. Read both off the real buildings in frame, then give each
new building a shadow of matching direction and a length proportionate to its own height -- a
taller one casts a longer shadow than a shorter one in the same frame. Include the shadow one
new building casts onto another, and remove any existing shadow the new buildings now block.

SURFACE
Give each the substance a real building has at this resolution: roof plant, services, lift
overruns and parapets on the roof; glazing lines, spandrels and material on the facades. Match
the grain, sharpness, colour balance and haze of the surrounding photograph. The edges where
they meet the ground, neighbouring roofs and the street should look photographed, not cut out.
They may differ from each other in age, material and colour, as neighbouring buildings do.

Neither {roof_word} nor {wall_word} may appear in the result. They are markings, not materials: the finished roof is the grey of membrane, metal deck or concrete with its plant and services, and the finished walls are glazing, precast or brick. A {roof_word} roof or a {wall_word} wall in the output means the marking was copied instead of read.

HOLD
Apart from the new buildings and the shadows they change, every pixel stays as it is. Add no
outline, label, annotation or watermark anywhere."""


REPLACE = """SCENE
This is an overhead satellite photograph of a site in Melbourne. A building stands on the
parcel now, and two things have been drawn over it. Outlined in {roof_word} and {wall_word}
lines, with the building still visible inside, is the building as it stands today -- about
{existing:.0f} m tall. Painted as a filled solid on the same footprint is what that same
building becomes after the alteration: {height:.0f} m tall, its {roof_word} face the new roof
and its {wall_word} faces the new sides. Those colours identify surfaces and are not the
building's materials.

TASK
Alter the building that is there so it matches the painted solid, and make the photograph
consistent with that alteration.

GEOMETRY
The altered building's ground position, footprint outline and roof outline are fixed as
painted. It is {height:.0f} m tall, about {storeys} storeys, on a footprint of {area:.0f}
square metres, land use {use}.

The one {roof_word} face is the new roof and the only roof. Every {wall_word} face is a
vertical wall seen from the side, not roof: put no plant, no services and no roof texture on
it. The roof is the {roof_word} outline, offset about {shift:.0f} m towards the {bearing} from
the ground outline, because a satellite {lean:.0f} degrees off vertical sees the top of a
{height:.0f} m building displaced that far. The ground outline is where the building meets the
ground.

Nothing of the building's former height may remain: no part of its old roof, no facade above
the new roofline, no rooftop plant left floating, and no trace of its old outline showing
through. Remove the marking lines themselves along with the height they mark -- they are
instructions, not part of the scene.

LIGHT
This is where the change shows most. The building was {existing:.0f} m and becomes
{height:.0f} m, so it was shading roughly {ratio:.1f} times as far as it now will. Find the
shadow the old building cast, work out how much of it a {height:.0f} m building would still
cast, and give back the rest: restore road surface, footpath,
neighbouring roof, vegetation and parked cars at the brightness those same surfaces have
elsewhere in this photograph where nothing shades them. Then cast the shortened shadow in the
direction every other shadow in the frame runs.

SURFACE
Give the altered building the substance a real one has at this resolution: roof plant,
services and parapets on its new roof; glazing lines and material on its facades. Ground and
neighbouring buildings that the taller building used to hide from view must be filled in from
what surrounds them -- continue the road, the footpath, the parking, the planting and the
adjacent roofs as they run on either side, at the grain, sharpness and colour balance of the
rest of the photograph.

Neither {roof_word} nor {wall_word} may appear in the result. They are markings, not materials: the finished roof is the grey of membrane, metal deck or concrete with its plant and services, and the finished walls are glazing, precast or brick. A {roof_word} roof or a {wall_word} wall in the output means the marking was copied instead of read.

HOLD
Apart from the altered building, what it used to hide and the shadows that change, every pixel
stays as it is. Add no outline, label, annotation or watermark anywhere."""


REMOVE = """SCENE
This is an overhead satellite photograph of a site in Melbourne. One building in it has been
outlined: a {wall_word} line around where it meets the ground, a {roof_word} line around its
roof, and {wall_word} lines up its visible corners joining the two. The building itself is
still visible inside that outline, unchanged. The lines mark which building is meant and how
tall it stands -- about {height:.0f} m, the two outlines being {shift:.0f} m apart towards the
{bearing} because the satellite is {lean:.0f} degrees off vertical.

TASK
Demolish the outlined building. The result should read as an ordinary satellite photograph of
the same place taken after demolition, with nothing standing on that parcel and no drawn lines
left anywhere.

GEOMETRY
The ground the building occupied becomes open. Do not put another building there, and do not
enlarge the neighbouring ones into the space. Buildings that share a wall with the one being
demolished keep their own walls, which now face open ground. Remove the marking lines
themselves along with the building -- they are instructions, not part of the scene.

LIGHT
The building's shadow goes with it. Find the shadow it casts -- it runs the same direction as
every other shadow in the frame and its length matches {height:.0f} m against the neighbours --
and restore whatever that shadow was lying on: road surface, footpath, neighbouring roof,
vegetation, parked cars, all at the brightness the unshaded parts of those same surfaces have
in this photograph. Light reaching ground that was in shade is the clearest sign the building
is gone.

SURFACE
Leave the cleared parcel as {ground}, rendered the way that surface looks from above in this
part of the city, at the grain, sharpness, colour balance and haze of the rest of the
photograph. Ground and neighbouring buildings the demolished building used to hide must be
filled in from what surrounds them -- continue the road, the footpath, the parking, the
planting and the adjacent roofs as they run on either side. Its edges against the street, the
footpath and the neighbouring buildings should look photographed, not cut out.

Neither {roof_word} nor {wall_word} may appear in the result -- the marking lines go with the
building they mark.

HOLD
Apart from the demolished building, what it used to hide and the shadows that change, every
pixel stays as it is. Add no outline, label, annotation or watermark anywhere."""


# what a demolished parcel is left as, offered in the panel so the operator picks rather than
# the model inventing
GROUNDS = ["hardstanding", "rubble and a levelled site", "a car park",
           "a fenced construction site", "grass"]

MODES = {
    "add": {"label": "new building", "prompt": ADD_ONE, "many": ADD_MANY,
            "paint": "solid", "needs_existing": False, "needs_ground": False},
    "replace": {"label": "alter what is there", "prompt": REPLACE, "many": None,
                "paint": "solid+wire", "needs_existing": True, "needs_ground": False},
    "remove": {"label": "demolish", "prompt": REMOVE, "many": None,
               "paint": "wire", "needs_existing": False, "needs_ground": True},
}
