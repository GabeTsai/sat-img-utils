def authenticate(project: str):
    import ee
    ee.Authenticate()
    ee.Initialize(project=project)


def main():
    pass

if __name__ == "__main__":
    main()