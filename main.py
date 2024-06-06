import pygame
import os
import math
import sys
import neat

#Screen dimensions
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700
SCREEN = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
TRACK = pygame.image.load(os.path.join("tracks/interlagos.png")).convert() #Can change track here
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH,SCREEN_HEIGHT-50))
clock = pygame.time.Clock()
CAR_SIZE_X = 9
CAR_SIZE_Y = 5

generation = 0

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("car.png")).convert()
        self.original_image = pygame.transform.scale(self.original_image, (CAR_SIZE_X,CAR_SIZE_Y))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH/2+100,160)) #Can change centre position of car sprites
        self.velocity = pygame.math.Vector2(1,0)
        self.position = self.rect.center
        self.angle = 180
        self.rotation_vel = 2
        
        self.direction = 0
        self.alive = True
        self.distance = 0
        self.prev_position = pygame.math.Vector2(0,0)
        self.radars = []

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60,-45,-30,0,30,45,60):
            self.radar(radar_angle)

        self.collision()
        
        self.sensor_info()

    def drive(self):
        self.prev_position = self.position
        self.position += self.velocity*0.5
        self.rect.center = self.position
        self.distance  = self.distance + (self.position - self.prev_position).magnitude()

    def rotate(self):
        if (self.direction == 1):
            self.angle -= self.rotation_vel
            
        if (self.direction == -1):
            self.angle += self.rotation_vel

        self.velocity = pygame.math.Vector2(1,0)
        self.velocity.rotate_ip(-self.angle)

        self.rect = self.image.get_rect(center = self.rect.center)
        self.image = pygame.transform.rotozoom(self.original_image, self.angle,1)


    def collision(self):
        length = 2
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle+20))*length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle+20))*length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle-20))*length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle-20))*length)]
        
        if self.inBounds(collision_point_right) and self.inBounds(collision_point_left):
            if (TRACK.get_at(collision_point_right) == pygame.Color(255, 255, 255) or
                TRACK.get_at(collision_point_left) == pygame.Color(255, 255, 255)):
                self.alive = False
        else:
            self.alive = False
            
            
        
        #pygame.draw.circle(SCREEN,(0,255,255,0), collision_point_right,1)
        #pygame.draw.circle(SCREEN,(0,255,255,0), collision_point_left,1)

    def inBounds(self, coords):
        return (0 <= coords[0] < SCREEN_WIDTH and 0 <= coords[1] < SCREEN_HEIGHT)


    
    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while 0<=x<SCREEN_WIDTH and 0<=y<SCREEN_HEIGHT-50 and not SCREEN.get_at((x,y)) == pygame.Color(255,255,255,255) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(360-self.angle - radar_angle)) * length)
            y = int(self.rect.center[1] + math.sin(math.radians(360-self.angle- radar_angle)) * length)
        
        

        pygame.draw.line(SCREEN, (0,161,155,100), self.rect.center, (x,y),1)
        

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x,2)
                             +math.pow(self.rect.center[1]-y,2)))
        self.radars.append([radar_angle,dist])

    def sensor_info(self):
        input = [0,0,0,0,0,0,0]
        for i , radar in enumerate(self.radars):
            input[i] = int(radar[1]/30)
        return input
    
    def reward(self):
        return self.distance / (CAR_SIZE_X / 2) 

def remove(index):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)


def run(genomes, config):
    pygame.init()
    global cars, ge, nets
    global generation
    cars = []
    ge = []
    nets = []
    
    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome,config)
        nets.append(net)
        genome.fitness = 0
    

    # global lap_counter

    while True:
        pygame.display.set_caption("AI cars go brrr")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        SCREEN.fill((255,255,255,255))
        SCREEN.blit(TRACK,(0,0))

        if len(cars) == 0:
            break

        for i, car in enumerate(cars):
            if car.sprite.alive and SCREEN.get_at((int(car.sprite.position[0]),int(car.sprite.position[1]))) != pygame.Color(255,255,255):
                ge[i].fitness += car.sprite.reward()
            if not car.sprite.alive:
                ge[i].fitness -= 50
                remove(i)

        for i,car in enumerate(cars):
            output = nets[i].activate(car.sprite.sensor_info())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0
        
        for car in cars:
            car.draw(SCREEN)
            car.sprite.update() 

        generation_font = pygame.font.SysFont("Arial", 30)
        text = generation_font.render("Generation: " + str(generation), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (SCREEN_WIDTH/2, 30)
        SCREEN.blit(text, text_rect)
        clock.tick(150)

        pygame.display.update()
        
        
    generation+=1

def main(config_path):
    global pop

    
   
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(run, 1000)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    main(config_path)
    

