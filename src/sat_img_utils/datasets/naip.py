import ee

def authenticate(project: str):
    ee.Authenticate()
    ee.Initialize(project=project)


def main():
    pass

if __name__ == "__main__":
    main()