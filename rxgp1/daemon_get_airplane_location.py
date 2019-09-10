from source.opensky_api import OpenSkyApi
from haversine import haversine


class AirPlane:
    def __init__(self, latitude, longitude, callsign, geo_altitude, on_ground, heading, to_location):
        self.latitude = latitude
        self.longitude = longitude
        self.callsign = callsign
        self.geo_altitude = geo_altitude
        self.on_ground = on_ground
        self.heading = heading
        self.to_location = to_location

    def __repr__(self):
        return repr(self.callsign + " / " + str(self.to_location))


if __name__ == '__main__':
    is_call_api = True
    center_location = (37.5572218, 126.792059) # Kimpo international airport
    while is_call_api:
        # bbox = (min latitude, max latitude, min longitude, max longitude)
        api = OpenSkyApi('rxgp1', 'tla0420!@')
        states = api.get_states(bbox=(34.3900458847, 38.6122429469, 126.117397903, 129.468304478)) # In Korea
        lst = []
        for s in states.states:
            airplane_location = (s.latitude, s.longitude)
            to_location = haversine(center_location, airplane_location)
            air_plane = AirPlane(latitude=s.latitude, longitude=s.longitude,
                     callsign=s.callsign, geo_altitude=s.geo_altitude,
                     on_ground=s.on_ground, heading=s.heading,
                     to_location=to_location)
            lst.append(air_plane)
        lst = sorted(lst, key=lambda x: x.to_location)
        print(lst)
